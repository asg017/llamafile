// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "llama.cpp/llama.h"
#include "llamafile/version.h"
#include "llama.cpp/embedr/sqlite3.h"
#include "llama.cpp/embedr/sqlite-vec.h"
#include "llama.cpp/embedr/sqlite-lembed.h"
#include "llama.cpp/embedr/sqlite-csv.h"
#include "llama.cpp/embedr/sqlite-lines.h"
#include "llama.cpp/embedr/shell.h"
#include <string.h>

#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cosmo.h>


int64_t time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

char * EMBEDR_MODEL = NULL;

int embedr_sqlite3_init(sqlite3 * db) {
  int rc;

  rc = sqlite3_vec_init(db, NULL, NULL); assert(rc == SQLITE_OK);
  rc = sqlite3_lembed_init(db, NULL, NULL); assert(rc == SQLITE_OK);
  rc = sqlite3_csv_init(db, NULL, NULL); assert(rc == SQLITE_OK);
  rc = sqlite3_lines_init(db, NULL, NULL); assert(rc == SQLITE_OK);

  if(!EMBEDR_MODEL) {
    return SQLITE_OK;
  }
  sqlite3_stmt * stmt;
  rc = sqlite3_prepare_v2(db, "insert into temp.lembed_models(model) values (?)", -1, &stmt, NULL);
  if(rc != SQLITE_OK) {
    assert(rc == SQLITE_OK);
    return rc;
  }
  sqlite3_bind_text(stmt, 1, EMBEDR_MODEL, -1, SQLITE_STATIC);
  sqlite3_step(stmt);
  rc = sqlite3_finalize(stmt);
  assert(rc == SQLITE_OK);

  return rc;
}

int table_exists(sqlite3 * db, const char * table) {
  int rc;
  sqlite3_stmt * stmt;
  rc = sqlite3_prepare_v2(db, "select ? in (select name from pragma_table_list)", -1, &stmt, NULL);
  assert(rc == SQLITE_OK);
  sqlite3_bind_text(stmt, 1, table, strlen(table), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  assert(rc == SQLITE_ROW);
  int result = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return result;
}

#define BAR_WIDTH 20
void print_progress_bar(long long nEmbed, long long nTotal, long long elapsed_ms) {
    float progress = (float)nEmbed / nTotal;
    int bar_fill = (int)(progress * BAR_WIDTH);

    long long remaining = nTotal - nEmbed;
    float rate = (float)nEmbed / (elapsed_ms / 1000.0);
    long long remaining_time = (rate > 0) ? remaining / rate : 0;

    printf("\r%3d%%|", (int)(progress * 100));
    for (int i = 0; i < BAR_WIDTH; i++) {
        if (i < bar_fill)
            printf("█");
        else
            printf(" ");
    }
    printf("| %lld/%lld [%02lld:%02lld<%02lld:%02lld, %.0f/s]",
           nEmbed, nTotal,
           elapsed_ms / 1000 / 60, elapsed_ms / 1000 % 60,
           remaining_time / 60, remaining_time % 60,
           rate);

    fflush(stdout);
}

int cmd_backfill(char * dbPath, char * table, char * column) {
  int rc;
  sqlite3* db;
  rc = sqlite3_open(dbPath, &db);
  if(rc != SQLITE_OK) {
    fprintf(stderr, "could not open database");
    return rc;
  }

  rc = embedr_sqlite3_init(db);
  assert(rc == SQLITE_OK);

  rc = sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  const char *tableEmbeddings = sqlite3_mprintf("%s_embeddings", table);
  assert(tableEmbeddings);

  if(!(table_exists(db, tableEmbeddings))) {
    const char * zSql = sqlite3_mprintf(
      "CREATE TABLE \"%w\"(rowid INTEGER PRIMARY KEY, embedding BLOB);"
      "INSERT INTO \"%w\"(rowid) SELECT rowid FROM \"%w\";"
      "CREATE INDEX \"idx_%w\" ON \"%w\"(embedding) WHERE embedding IS NULL;",
      tableEmbeddings,
      tableEmbeddings,
      table,
      tableEmbeddings,
      tableEmbeddings
    );
    rc = sqlite3_exec(db, zSql, NULL, NULL, NULL);
    sqlite3_free((void *) zSql);
    assert(rc == SQLITE_OK);
  }


  int64_t nTotal;
  {
    sqlite3_stmt * stmt;
    const char * zSql = sqlite3_mprintf("SELECT count(*) FROM \"%w\" WHERE embedding IS NULL", tableEmbeddings);
    assert(zSql);
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    assert(rc == SQLITE_OK);
    rc = sqlite3_step(stmt);
    assert(rc == SQLITE_ROW);
    nTotal = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
  }

  int64_t nRemaining = nTotal;


  sqlite3_stmt * stmt;
  const char * zSql = sqlite3_mprintf(
    " \
      WITH chunk AS ( \
        SELECT \
          e.rowid, \
          lembed(\"%w\") AS embedding \
        FROM \"%w\" AS e \
        LEFT JOIN \"%w\" AS src ON src.rowid = e.rowid \
        WHERE e.embedding IS NULL \
        LIMIT ? \
      ) \
      UPDATE \"%w\" AS e \
      SET embedding = chunk.embedding \
      FROM chunk \
      WHERE e.rowid = chunk.rowid \
      RETURNING rowid \
    ",
    column,
    tableEmbeddings,
    table,
    tableEmbeddings
  );
  assert(zSql);

  rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
  sqlite3_free((void *) zSql);
  assert(rc == SQLITE_OK);

  sqlite3_bind_int(stmt, 1, 16);

  int64_t nEmbed = 0;
  int64_t t0 = time_ms();

  while(1){
    sqlite3_reset(stmt);

    int nChunkEmbed = 0;
    while(1) {
      rc = sqlite3_step(stmt);
      if(rc == SQLITE_DONE) {
        break;
      }
      assert(rc == SQLITE_ROW);
      nChunkEmbed++;
    }
    if(nChunkEmbed == 0) {
      break;
    }
    nEmbed += nChunkEmbed;
    nRemaining -= nChunkEmbed;
    print_progress_bar(nEmbed, nTotal, time_ms() - t0);
  }




  rc = sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  sqlite3_free((void *) tableEmbeddings);
  sqlite3_close(db);
  return 0;
}

int cmd_embed(char * source) {
  int rc;
  sqlite3* db;
  sqlite3_stmt * stmt;

  rc = sqlite3_open(":memory:", &db);
  assert(rc == SQLITE_OK);

  rc = embedr_sqlite3_init(db);
  assert(rc == SQLITE_OK);

  rc = sqlite3_prepare_v2(db, "select vec_to_json(lembed(?))", -1, &stmt, NULL);
  assert(rc == SQLITE_OK);

  sqlite3_bind_text(stmt, 1, source, strlen(source), SQLITE_STATIC);

  rc = sqlite3_step(stmt);
  assert(rc == SQLITE_ROW);

  printf("%.*s", sqlite3_column_bytes(stmt, 0), sqlite3_column_text(stmt, 0));

  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return 0;
}


int cmd_sh(int argc, char * argv[]) {
  return mn(argc, argv);
}

int main(int argc, char ** argv) {
    int rc;
    sqlite3* db;
    sqlite3_stmt* stmt;

    FLAG_log_disable = 1;
    argc = cosmo_args("/zip/.args", &argv);
    FLAGS_READY = 1;

    char * modelPath = NULL;
    for(int i = 1; i < argc; i++) {
      char * arg = argv[i];
      if(sqlite3_stricmp(arg, "--model") == 0 || sqlite3_stricmp(arg, "-m") == 0) {
        assert(++i <= argc);
        EMBEDR_MODEL = argv[i];
      }
      else if(sqlite3_stricmp(arg, "--version") == 0 || sqlite3_stricmp(arg, "-v") == 0) {
        fprintf(stderr,
          "llamafile-embed %s, SQLite %s, sqlite-vec=%s, sqlite-lembed=%s\n",
          LLAMAFILE_VERSION_STRING,
          sqlite3_version,
          SQLITE_VEC_VERSION,
          SQLITE_LEMBED_VERSION
        );
        return 0;
      }
      else if(sqlite3_stricmp(arg, "sh") == 0) {
        return cmd_sh(argc-i, argv+i);
      }
      else if(sqlite3_stricmp(arg, "embed") == 0) {
        assert(i + 2 == argc);
        return cmd_embed(argv[i+1]);
      }
      else if(sqlite3_stricmp(arg, "backfill") == 0) {
        assert(i + 5 == argc);
        char * dbpath = argv[i+1];
        char * table = argv[i+2];
        char * column = argv[i+3];
        return cmd_backfill(dbpath, table, column);
      }
      else {
        printf("Unknown arg %s\n", arg);
        return 1;
      }
    }

    return 0;
}
