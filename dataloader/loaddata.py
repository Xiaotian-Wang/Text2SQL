import json
import pickle
import networkx as nx
import attr
import re
import torch
from typing import List, Dict
from pathlib import Path
import sqlite3
from tqdm import tqdm
from datasets.spider_lib import evaluation

@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()
    connection = attr.ib(default=None)


def postprocess_original_name(s: str):
    return re.sub(r'([A-Z]+)', r' \1', s).replace('_', ' ').lower().strip()


def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def load_tables(paths):
    schemas = {}
    eval_foreign_key_maps = {}

    for path in paths:
        schema_dicts = json.load(open(path, encoding='utf8'))
        for schema_dict in schema_dicts:
            tables = tuple(
                Table(
                    id=i,
                    name=name.split(),
                    unsplit_name=name,
                    orig_name=orig_name,
                )
                for i, (name, orig_name) in enumerate(zip(
                    schema_dict['table_names'], schema_dict['table_names_original']))
            )
            columns = tuple(
                Column(
                    id=i,
                    table=tables[table_id] if table_id >= 0 else None,
                    name=col_name.split(),
                    unsplit_name=col_name,
                    orig_name=orig_col_name,
                    type=col_type,
                )
                for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                    schema_dict['column_names'],
                    schema_dict['column_names_original'],
                    schema_dict['column_types']))
            )

            # Link columns to tables
            for column in columns:
                if column.table:
                    column.table.columns.append(column)

            for column_id in schema_dict['primary_keys']:
                # Register primary keys
                column = columns[column_id]
                column.table.primary_keys.append(column)

            foreign_key_graph = nx.DiGraph()
            for source_column_id, dest_column_id in schema_dict['foreign_keys']:
                # Register foreign keys
                source_column = columns[source_column_id]
                dest_column = columns[dest_column_id]
                source_column.foreign_key_for = dest_column
                foreign_key_graph.add_edge(
                    source_column.table.id,
                    dest_column.table.id,
                    columns=(source_column_id, dest_column_id))
                foreign_key_graph.add_edge(
                    dest_column.table.id,
                    source_column.table.id,
                    columns=(dest_column_id, source_column_id))

            db_id = schema_dict['db_id']
            assert db_id not in schemas
            schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
            eval_foreign_key_maps[db_id] = build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps


class SpiderDataset(torch.utils.data.Dataset):
    def __init__(self, paths, tables_paths, db_path, demo_path=None, limit=None):
        self.paths = paths
        self.db_path = db_path
        self.examples = []

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)

        for path in paths:
            raw_data = json.load(open(path, encoding='utf8'))
            for entry in raw_data:
                item = SpiderItem(
                    text=entry['question'].split(),
                    code=entry['sql'],
                    schema=self.schemas[entry['db_id']],
                    orig=entry,
                    orig_schema=self.schemas[entry['db_id']].orig)
                self.examples.append(item)

        if demo_path:
            self.demos: Dict[str, List] = json.load(open(demo_path))

        # Backup in-memory copies of all the DBs and create the live connections
        for db_id, schema in tqdm(self.schemas.items(), desc="DB connections"):
            sqlite_path = Path(db_path) / db_id / f"{db_id}.sqlite"
            source: sqlite3.Connection
            with sqlite3.connect(str(sqlite_path)) as source:
                dest = sqlite3.connect(':memory:')
                dest.row_factory = sqlite3.Row
                # source.backup(dest)
            schema.connection = dest

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __del__(self):
        for _, schema in self.schemas.items():
            if schema.connection:
                schema.connection.close()

    class Metrics:
        def __init__(self, dataset):
            self.dataset = dataset
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = evaluation.Evaluator(
                self.dataset.db_path,
                self.foreign_key_maps,
                'match')
            self.results = []

        def add(self, item, inferred_code, orig_question=None):
            ret_dict = self.evaluator.evaluate_one(
                item.schema.db_id, item.orig['query'], inferred_code)
            if orig_question:
                ret_dict["orig_question"] = orig_question
            self.results.append(ret_dict)

        def add_beams(self, item, inferred_codes, orig_question=None):
            beam_dict = {}
            if orig_question:
                beam_dict["orig_question"] = orig_question
            for i, code in enumerate(inferred_codes):
                ret_dict = self.evaluator.evaluate_one(
                    item.schema.db_id, item.orig['query'], code)
                beam_dict[i] = ret_dict
                if ret_dict["exact"] is True:
                    break
            self.results.append(beam_dict)

        def finalize(self):
            self.evaluator.finalize()
            return {
                'per_item': self.results,
                'total_scores': self.evaluator.scores
            }


def _linking_wrapper(fn_linking):
    """wrapper for linking function, do linking and id convert

    Args:
        fn_linking (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    link_result = fn_linking(self.question_tokens, self.db)

    # convert words id to BERT word pieces id
    new_result = {}
    for m_name, matches in link_result.items():
        new_match = {}
        for pos_str, match_type in matches.items():
            qid_str, col_tab_id_str = pos_str.split(',')
            qid, col_tab_id = int(qid_str), int(col_tab_id_str)
            for real_qid in self.token_mapping[qid]:
                new_match[f'{real_qid},{col_tab_id}'] = match_type
        new_result[m_name] = new_match
    return new_result


if __name__ == "__main__":
    data_path = '../data/CSpider/database'
    data_schema_path = '../data/CSpider/db_schema.json'
    a = load_tables([data_schema_path])
    data1 = SpiderDataset(paths=['../data/CSpider/dev.json'], tables_paths=[data_schema_path], db_path=data_path)

