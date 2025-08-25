// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "embedfile/embedfile.h"
#include "embedfile/shell.h"
#include "embedfile/sqlite-csv.h"
#include "embedfile/sqlite-lembed.h"
#include "embedfile/sqlite-lines.h"
#include "embedfile/sqlite-vec.h"
#include "llama.cpp/llama.h"
#include "llamafile/version.h"
#include "third_party/sqlite/sqlite3.h"
#include <string.h>

#include <cosmo.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_SQLITE_NOT_OK(rc, db) \
    do { \
        if ((rc) != SQLITE_OK) { \
            fprintf(stderr, "SQLite error: %s\n", sqlite3_errmsg(db)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#define CHECK_SQLITE_NOT_DONE(rc, db) \
    do { \
        if ((rc) != SQLITE_DONE) { \
            fprintf(stderr, "SQLite error: %s\n", sqlite3_errmsg(db)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#define CHECK_SQLITE_NOT_ROW(rc, db) \
    do { \
        if ((rc) != SQLITE_ROW) { \
            fprintf(stderr, "SQLite error: %s\n", sqlite3_errmsg(db)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#define CHECK_ZSQL_NOT_NULL(zSql) \
    do { \
        if (!(zSql)) { \
            fprintf(stderr, "Error: Cannot create SQL statement.\n"); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int64_t time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

char *EMBEDFILE_MODEL = NULL;

void embedfile_version(sqlite3_context *context, int argc, sqlite3_value **value) {
    sqlite3_result_text(context, EMBEDFILE_VERSION, -1, SQLITE_STATIC);
}

int embedfile_sqlite3_init(sqlite3 *db) {
    int rc;

    rc = sqlite3_vec_init(db, NULL, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_lembed_init(db, NULL, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_csv_init(db, NULL, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_lines_init(db, NULL, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_create_function_v2(db, "embedfile_version", 0, SQLITE_DETERMINISTIC | SQLITE_UTF8,
                                    NULL, embedfile_version, NULL, NULL, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);

    if (!EMBEDFILE_MODEL) {
        fprintf(stderr, "Error: No model provided.\n");
        exit(EXIT_FAILURE);
    }
    sqlite3_stmt *stmt;
    rc =
        sqlite3_prepare_v2(db, "insert into temp.lembed_models(model) values (?)", -1, &stmt, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_bind_text(stmt, 1, EMBEDFILE_MODEL, -1, SQLITE_STATIC);
    CHECK_SQLITE_NOT_OK(rc, db);
    sqlite3_step(stmt);
    rc = sqlite3_finalize(stmt);
    CHECK_SQLITE_NOT_OK(rc, db);
    return rc;
}

int table_exists(sqlite3 *db, const char *table) {
    int rc;
    sqlite3_stmt *stmt;
    rc =
        sqlite3_prepare_v2(db, "select ? in (select name from pragma_table_list)", -1, &stmt, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_bind_text(stmt, 1, table, strlen(table), SQLITE_STATIC);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_step(stmt);
    CHECK_SQLITE_NOT_ROW(rc, db);
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
    printf("| %lld/%lld [%02lld:%02lld<%02lld:%02lld, %.0f/s]", nEmbed, nTotal,
           elapsed_ms / 1000 / 60, elapsed_ms / 1000 % 60, remaining_time / 60, remaining_time % 60,
           rate);

    fflush(stdout);
}

int default_model_dimensions(sqlite3 *db, int64_t *dimensions) {
    int rc;
    sqlite3_stmt *stmt;
    rc = sqlite3_prepare_v2(db, "select dimensions from lembed_models where name = ?", -1, &stmt,
                            NULL);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_bind_text(stmt, 1, "default", -1, SQLITE_STATIC);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_step(stmt);
    if(rc == SQLITE_DONE) {
      return SQLITE_EMPTY;
    }
    CHECK_SQLITE_NOT_ROW(rc, db);
    *dimensions = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);

    return SQLITE_OK;
}

int cmd_search(char *dbPath, char *query, int k) {
    int rc;
    sqlite3 *db;
    sqlite3_stmt *stmt;
    rc = sqlite3_open(dbPath, &db);
    CHECK_SQLITE_NOT_OK(rc, db);

    rc = embedfile_sqlite3_init(db);
    CHECK_SQLITE_NOT_OK(rc, db);

    rc = sqlite3_prepare_v2(db, "\
    SELECT \
      substr(name, 1, instr(name, '_') - 1) AS source_column, \
      name AS embbedding_column \
    FROM pragma_table_xinfo('vec_items')  \
    WHERE name LIKE '%_embedding'; \
    ",
                            -1, &stmt, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);

    rc = sqlite3_step(stmt);
    CHECK_SQLITE_NOT_ROW(rc, db);
    char *sourceColumn = sqlite3_mprintf("%s", sqlite3_column_text(stmt, 0));
    char *embeddingsColumn = sqlite3_mprintf("%s", sqlite3_column_text(stmt, 1));
    sqlite3_finalize(stmt);

    const char *zSql = sqlite3_mprintf("                               \
      SELECT                        \
        vec_items.rowid,                      \
        items.\"%w\",  \
        vec_items.distance                    \
      FROM vec_items \
      LEFT JOIN items ON items.rowid = vec_items.rowid \
      WHERE \"%w\" MATCH lembed(?)  \
      AND k = ?   \
    ",
                                       sourceColumn, embeddingsColumn);
    CHECK_ZSQL_NOT_NULL(zSql);
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    sqlite3_free((void *)zSql);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_bind_text(stmt, 1, query, strlen(query), SQLITE_STATIC);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_bind_int64(stmt, 2, k);
    CHECK_SQLITE_NOT_OK(rc, db);

    while (1) {
        rc = sqlite3_step(stmt);
        if (rc == SQLITE_DONE) {
            break;
        }
        CHECK_SQLITE_NOT_ROW(rc, db);

        printf("%lld %f %.*s\n", sqlite3_column_int64(stmt, 0), sqlite3_column_double(stmt, 2),
               sqlite3_column_bytes(stmt, 1), sqlite3_column_text(stmt, 1));
    }

    sqlite3_finalize(stmt);

    sqlite3_free(sourceColumn);
    sqlite3_free(embeddingsColumn);
    sqlite3_close(db);
    return 0;
}

int cmd_embed(char *source) {
    int rc;
    sqlite3 *db;
    sqlite3_stmt *stmt;

    rc = sqlite3_open(":memory:", &db);
    CHECK_SQLITE_NOT_OK(rc, db);

    rc = embedfile_sqlite3_init(db);
    CHECK_SQLITE_NOT_OK(rc, db);

    if (source) {
        rc = sqlite3_prepare_v2(db, "select vec_to_json(lembed(?))", -1, &stmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_bind_text(stmt, 1, source, strlen(source), SQLITE_STATIC);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_step(stmt);
        CHECK_SQLITE_NOT_ROW(rc, db);

        printf("%.*s", sqlite3_column_bytes(stmt, 0), sqlite3_column_text(stmt, 0));
    } else {
        rc = sqlite3_prepare_v2(
            db, "select vec_to_json(lembed(line)) from lines_read('/dev/stdin')", -1, &stmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);

        while (1) {
            rc = sqlite3_step(stmt);
            if (rc == SQLITE_DONE) {
                break;
            }
            CHECK_SQLITE_NOT_ROW(rc, db);
            printf("%.*s", sqlite3_column_bytes(stmt, 0), sqlite3_column_text(stmt, 0));
        }
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return 0;
}

typedef enum {
    EF_IMPORT_SOURCE_TYPE_CSV,
    EF_IMPORT_SOURCE_TYPE_JSON,
    EF_IMPORT_SOURCE_TYPE_NDJSON,
    EF_IMPORT_SOURCE_TYPE_TXT,
    EF_IMPORT_SOURCE_TYPE_DB
} ef_import_source_type;

struct aux {
    sqlite3_stmt *monitorStmt;
    sqlite3_stmt *stmt;
    int64_t total;
    int64_t started_at;
};

int progress(void *p) {
    struct aux *x = p;
    int rc = sqlite3_bind_pointer(x->monitorStmt, 1, x->stmt, "stmt-pointer", 0);
    if (rc != SQLITE_OK) {
        return 0;
    }
    while (1) {
        int rc = sqlite3_step(x->monitorStmt);
        if (rc == SQLITE_DONE) {
            break;
        }
        if (rc != SQLITE_ROW) {
            sqlite3_reset(x->monitorStmt);
            return 0;
        }
        print_progress_bar(sqlite3_column_int64(x->monitorStmt, 8), x->total,
                           time_ms() - x->started_at);
    }
    sqlite3_reset(x->monitorStmt);
    return 0;
}

int monitor_stmt(sqlite3_stmt *stmt, int64_t total) {
    int rc;
    sqlite3_stmt *monitorStmt;
    const char *sql = "  SELECT addr, opcode, p1, p2, p3, p4, p5, comment, nexec,"
                      "   round(ncycle*100.0 / (sum(ncycle) OVER ()), 2)||'%' AS cycles"
                      "   FROM bytecode(?)"
                      "WHERE opcode = 'VUpdate'";
    rc = sqlite3_prepare_v2(sqlite3_db_handle(stmt), sql, -1, &monitorStmt, NULL);
    CHECK_SQLITE_NOT_OK(rc, sqlite3_db_handle(stmt));
    struct aux x = {.monitorStmt = monitorStmt, .stmt = stmt, .total = total};
    x.started_at = time_ms();
    sqlite3_progress_handler(sqlite3_db_handle(stmt), 1000, progress, &x);
    sqlite3_step(stmt);
    sqlite3_progress_handler(sqlite3_db_handle(stmt), 0, NULL, NULL);

    sqlite3_finalize(monitorStmt);

    return sqlite3_finalize(stmt);
}

int cmd_import(int argc, char *argv[]) {
    char *embedColumn = NULL;
    ef_import_source_type source_type = EF_IMPORT_SOURCE_TYPE_TXT;
    char *srcFile = NULL;
    char *indexFile = NULL;
    char *table = NULL;

    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];
        if (sqlite3_stricmp(arg, "--embed") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing value for --embed.\n");
                exit(EXIT_FAILURE);
            }
            embedColumn = argv[i];
        } else if (sqlite3_stricmp(arg, "--table") == 0 || sqlite3_stricmp(arg, "-t") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing value for --table.\n");
                exit(EXIT_FAILURE);
            }
            table = argv[i];
        } else {
            if (!srcFile) {
                srcFile = argv[i];
            } else if (!indexFile) {
                indexFile = argv[i];
            } else {
                fprintf(stderr, "Error: Unknown extra argument %s.\n", argv[i]);
                return 1;
            }
        }
    }
    if (!srcFile) {
        fprintf(stderr, "Error: No source file provided.\n");
        exit(EXIT_FAILURE);
    }
    if (!indexFile) {
        fprintf(stderr, "Error: No index db file provided.\n");
        exit(EXIT_FAILURE);
    }

    if (sqlite3_strlike("%csv", srcFile, 0) == 0) {
        source_type = EF_IMPORT_SOURCE_TYPE_CSV;
    } else if (sqlite3_strlike("%ndjson", srcFile, 0) == 0) {
        source_type = EF_IMPORT_SOURCE_TYPE_NDJSON;
    } else if (sqlite3_strlike("%json", srcFile, 0) == 0) {
        source_type = EF_IMPORT_SOURCE_TYPE_JSON;
    } else if (sqlite3_strlike("%txt", srcFile, 0) == 0) {
        source_type = EF_IMPORT_SOURCE_TYPE_TXT;
    } else if (sqlite3_strlike("%db", srcFile, 0) == 0) {
        source_type = EF_IMPORT_SOURCE_TYPE_DB;
        if (!table) {
            fprintf(stderr, "Error: No table provided.\n");
            exit(EXIT_FAILURE);
        }
    } else {
        fprintf(stderr, "Error: Couldn't determine type of source file %s\n", srcFile);
    }

    if (source_type == EF_IMPORT_SOURCE_TYPE_TXT) {
        embedColumn = "line";
    } else {
        if (!embedColumn) {
            fprintf(stderr, "Error: No embed column provided.\n");
            exit(EXIT_FAILURE);
        }
    }

    int rc;
    sqlite3 *db;
    rc = sqlite3_open(indexFile, &db);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = embedfile_sqlite3_init(db);
    CHECK_SQLITE_NOT_OK(rc, db);

    rc = sqlite3_fileio_init(db, NULL, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);

    sqlite3_stmt *stmt;
    const char *zSql = NULL;
    switch (source_type) {
    case EF_IMPORT_SOURCE_TYPE_CSV: {
        zSql = sqlite3_mprintf(
            "CREATE VIRTUAL TABLE temp.source USING csv(filename=\"%w\", header=yes)", srcFile);
        CHECK_ZSQL_NOT_NULL(zSql);
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        break;
    }
    case EF_IMPORT_SOURCE_TYPE_JSON: {
        sqlite3_stmt *innerStmt;
        rc = sqlite3_prepare_v2(
            db, "select fullkey, key from json_each(json_extract(readfile(?), '$[0]'))", -1,
            &innerStmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_bind_text(innerStmt, 1, srcFile, strlen(srcFile), SQLITE_STATIC);
        CHECK_SQLITE_NOT_OK(rc, db);

        sqlite3_str *sqlStr = sqlite3_str_new(NULL);
        sqlite3_str_appendf(sqlStr, "CREATE TABLE  temp.source AS SELECT rowid");
        while (1) {
            rc = sqlite3_step(innerStmt);
            if (rc == SQLITE_DONE) {
                break;
            }
            CHECK_SQLITE_NOT_ROW(rc, db);
            sqlite3_str_appendf(sqlStr, ", value ->> %Q as \"%w\"",
                                sqlite3_column_text(innerStmt, 0),
                                sqlite3_column_text(innerStmt, 1));
        }
        sqlite3_finalize(innerStmt);

        sqlite3_str_appendf(sqlStr, " FROM json_each(readfile(?))");
        zSql = sqlite3_str_finish(sqlStr);
        CHECK_ZSQL_NOT_NULL(zSql);
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_bind_text(stmt, 1, srcFile, strlen(srcFile), SQLITE_STATIC);
        CHECK_SQLITE_NOT_OK(rc, db);
        break;
    }
    case EF_IMPORT_SOURCE_TYPE_NDJSON: {
        sqlite3_stmt *innerStmt;
        rc = sqlite3_prepare_v2(
            db, "select fullkey, key from json_each((select line from lines_read(?) limit 1))", -1,
            &innerStmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_bind_text(innerStmt, 1, srcFile, strlen(srcFile), SQLITE_STATIC);
        CHECK_SQLITE_NOT_OK(rc, db);

        sqlite3_str *sqlStr = sqlite3_str_new(NULL);
        sqlite3_str_appendf(sqlStr, "CREATE TABLE  temp.source AS SELECT rowid");
        while (1) {
            rc = sqlite3_step(innerStmt);
            if (rc == SQLITE_DONE) {
                break;
            }
            CHECK_SQLITE_NOT_ROW(rc, db);
            sqlite3_str_appendf(sqlStr, ", line ->> %Q as \"%w\"",
                                sqlite3_column_text(innerStmt, 0),
                                sqlite3_column_text(innerStmt, 1));
        }
        sqlite3_finalize(innerStmt);

        sqlite3_str_appendf(sqlStr, " FROM lines_read(?)");
        zSql = sqlite3_str_finish(sqlStr);
        CHECK_ZSQL_NOT_NULL(zSql);
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_bind_text(stmt, 1, srcFile, strlen(srcFile), SQLITE_STATIC);
        CHECK_SQLITE_NOT_OK(rc, db);
        break;
    }
    case EF_IMPORT_SOURCE_TYPE_TXT: {
        zSql =
            sqlite3_mprintf("CREATE TABLE temp.source AS SELECT line FROM lines_read(?)", srcFile);
        CHECK_ZSQL_NOT_NULL(zSql);
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_bind_text(stmt, 1, srcFile, strlen(srcFile), SQLITE_STATIC);
        CHECK_SQLITE_NOT_OK(rc, db);
        break;
    }
    case EF_IMPORT_SOURCE_TYPE_DB: {
        sqlite3 *srcDb;
        rc = sqlite3_open(srcFile, &srcDb);
        CHECK_SQLITE_NOT_OK(rc, srcDb);

        sqlite3_stmt *checkStmt;
        rc = sqlite3_prepare_v2(srcDb,
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", -1, &checkStmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, srcDb);
        sqlite3_bind_text(checkStmt, 1, table, -1, SQLITE_STATIC);
        rc = sqlite3_step(checkStmt);
        CHECK_SQLITE_NOT_ROW(rc, srcDb);
        sqlite3_finalize(checkStmt);

        zSql = sqlite3_mprintf("ATTACH DATABASE %Q AS srcdb;", srcFile);
        CHECK_ZSQL_NOT_NULL(zSql);
        rc = sqlite3_exec(db, zSql, NULL, NULL, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        sqlite3_free((void *)zSql);
    
        zSql = sqlite3_mprintf("CREATE TABLE temp.source AS SELECT * FROM srcdb.%Q", table);
        CHECK_ZSQL_NOT_NULL(zSql);
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        sqlite3_free((void *)zSql);
        rc = sqlite3_step(stmt);
        CHECK_SQLITE_NOT_DONE(rc, db);
        sqlite3_finalize(stmt);
    
        zSql = sqlite3_mprintf("DETACH DATABASE srcdb;");
        CHECK_ZSQL_NOT_NULL(zSql);
        rc = sqlite3_exec(db, zSql, NULL, NULL, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        sqlite3_free((void *)zSql);
        break;
    }
    default: {
        fprintf(stderr, "Error: Bad import source type.\n");
        exit(EXIT_FAILURE);
    }
    }

    rc = sqlite3_step(stmt);
    CHECK_SQLITE_NOT_DONE(rc, db);
    sqlite3_finalize(stmt);
    sqlite3_free((void *)zSql);

    if (!table_exists(db, "vec_items")) {
        int64_t dimensions;
        rc = default_model_dimensions(db, &dimensions);
        if(rc == SQLITE_EMPTY) {
          fprintf(stderr, "ERROR: No embeddings model registed with embedfile yet. Try with the `--model path/to/model.gguf` flag.");
          return 1;
        }
        CHECK_SQLITE_NOT_OK(rc, db);

        zSql =
            sqlite3_mprintf("CREATE VIRTUAL TABLE vec_items USING vec0( %w_embedding float[%lld])",
                            embedColumn, dimensions);
        CHECK_ZSQL_NOT_NULL(zSql);
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
        sqlite3_free((void *)zSql);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_step(stmt);
        CHECK_SQLITE_NOT_DONE(rc, db);
        sqlite3_finalize(stmt);
    }

    if (!table_exists(db, "items")) {
        rc = sqlite3_prepare_v2(db, "CREATE TABLE items AS SELECT * FROM temp.source LIMIT 0;", -1,
                                &stmt, NULL);
        CHECK_SQLITE_NOT_OK(rc, db);
        rc = sqlite3_step(stmt);
        CHECK_SQLITE_NOT_DONE(rc, db);
        sqlite3_finalize(stmt);
    }

    zSql = sqlite3_mprintf("INSERT INTO items SELECT * FROM temp.source;", embedColumn);
    CHECK_ZSQL_NOT_NULL(zSql);
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    sqlite3_free((void *)zSql);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_step(stmt);
    CHECK_SQLITE_NOT_DONE(rc, db);
    sqlite3_finalize(stmt);

    zSql = sqlite3_mprintf("SELECT COUNT(*) from temp.source;", embedColumn);
    CHECK_ZSQL_NOT_NULL(zSql);
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    sqlite3_free((void *)zSql);
    CHECK_SQLITE_NOT_OK(rc, db);
    rc = sqlite3_step(stmt);
    CHECK_SQLITE_NOT_ROW(rc, db);
    int64_t n = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);

    rc = sqlite3_exec(db, "SELECT * FROM temp.source", NULL, NULL, NULL);
    CHECK_SQLITE_NOT_OK(rc, db);

    zSql = sqlite3_mprintf("INSERT INTO vec_items SELECT rowid, lembed(\"%w\") FROM temp.source;",
                           embedColumn);
    CHECK_ZSQL_NOT_NULL(zSql);
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    sqlite3_free((void *)zSql);
    CHECK_SQLITE_NOT_OK(rc, db);
#define LIGHT_BLUE "\033[1;34m"
#define GREEN "\033[0;32m"
#define RESET "\033[0m"
#define LIGHT_GREY "\033[0;37m"
#define MAGENTA "\033[0;35m"
    printf(LIGHT_BLUE "%s" RESET "\n", sqlite3_expanded_sql(stmt));

    rc = monitor_stmt(stmt, n);
    CHECK_SQLITE_NOT_OK(rc, db);

    printf(GREEN "\u2714" RESET " %s imported into %s, %d items\n", srcFile, indexFile,
           sqlite3_changes(db));

    sqlite3_close(db);
    return 0;
}

int cmd_sh(int argc, char *argv[]) {
    return mn(argc, argv);
}

int main(int argc, char **argv) {
    int rc;
    sqlite3 *db;
    sqlite3_stmt *stmt;

    FLAG_log_disable = 1;
    argc = cosmo_args("/zip/.args", &argv);
    FLAGS_READY = 1;

    char *modelPath = NULL;
    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];
        if (sqlite3_stricmp(arg, "--model") == 0 || sqlite3_stricmp(arg, "-m") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing value for --model.\n");
                exit(EXIT_FAILURE);
            }
            EMBEDFILE_MODEL = argv[i];
        } else if (sqlite3_stricmp(arg, "--version") == 0 || sqlite3_stricmp(arg, "-v") == 0) {
            fprintf(stderr,
                    "embedfile %s, llamafile %s, SQLite %s, sqlite-vec=%s, sqlite-lembed=%s\n",
                    EMBEDFILE_VERSION, LLAMAFILE_VERSION_STRING, sqlite3_version,
                    SQLITE_VEC_VERSION, SQLITE_LEMBED_VERSION);
            return 0;
        } else if (sqlite3_stricmp(arg, "--help") == 0 || sqlite3_stricmp(arg, "-h") == 0) {
            llamafile_help("/zip/embedfile/embedfile.1.asc");
            fprintf(stderr,
                    "embedfile %s, llamafile %s, SQLite %s, sqlite-vec=%s, sqlite-lembed=%s\n",
                    EMBEDFILE_VERSION, LLAMAFILE_VERSION_STRING, sqlite3_version,
                    SQLITE_VEC_VERSION, SQLITE_LEMBED_VERSION);
            fprintf(stderr, "Usage: embedfile [sh,embed,backfill,index]\n");
            return 0;
        } else if (sqlite3_stricmp(arg, "sh") == 0) {
            return cmd_sh(argc - i, argv + i);
        } else if (sqlite3_stricmp(arg, "embed") == 0) {
            return cmd_embed(i + 2 == argc ? argv[i + 1] : NULL);
        } else if (sqlite3_stricmp(arg, "search") == 0) {
            char *dbpath = NULL;
            char *query = NULL;
            int k = 10;
            for (int j = i + 1; j < argc; j++) {
                if (sqlite3_stricmp(argv[j], "--k") == 0 || sqlite3_stricmp(argv[j], "-k") == 0) {
                    if (j + 1 >= argc) {
                        fprintf(stderr, "Error: Missing value for --k.\n");
                        exit(EXIT_FAILURE);
                    }
                    k = atoi(argv[++j]);
                    if (k <= 0) {
                        fprintf(stderr, "Error: --k must be a positive integer.\n");
                        return 1;
                    }
                } else if (!dbpath) {
                    dbpath = argv[j];
                } else if (!query) {
                    query = argv[j];
                } else {
                    fprintf(stderr, "Error: Unknown extra argument %s.\n", argv[j]);
                    exit(EXIT_FAILURE);
                }
            }
            if (!dbpath || !query) {
                fprintf(stderr, "Error: No database nor search query provided.\n");
                exit(EXIT_FAILURE);
            }
            return cmd_search(dbpath, query, k);
        } else if (sqlite3_stricmp(arg, "import") == 0) {
            return cmd_import(argc - i, argv + i);
        } else {
            fprintf(stderr, "Error: Unknown argument %s.\n", arg);
            exit(EXIT_FAILURE);
        }
    }

    fprintf(stderr, "Usage: embedfile [--model MODEL_FILE] [embed [TEXT] | sh | import [--embed COLUMN] [--table NAME] SOURCE_FILE INDEX_DB | search [--k NUM] INDEX_DB QUERY]\n");
    return 0;
}
