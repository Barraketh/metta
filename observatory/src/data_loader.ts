import * as duckdb from '@duckdb/duckdb-wasm';
import * as arrow from 'apache-arrow';
import duckdb_wasm from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import mvp_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';
import duckdb_wasm_eh from '@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url';
import eh_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url';

const dbUri = import.meta.env.VITE_EVAL_DB_URI;

async function initDuckDB() {
  const MANUAL_BUNDLES: duckdb.DuckDBBundles = {
    mvp: {
        mainModule: duckdb_wasm,
        mainWorker: mvp_worker,
    },
    eh: {
        mainModule: duckdb_wasm_eh,
        mainWorker: eh_worker,
    },
  };
  // Select a bundle based on browser checks
  const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);
  // Instantiate the asynchronous version of DuckDB-wasm
  const worker = new Worker(bundle.mainWorker!);
  const logger = new duckdb.ConsoleLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);

  const conn = await db.connect()
  await conn.send(`ATTACH DATABASE '${dbUri}' AS eval`);
  conn.close();

  return db
}

const db = await initDuckDB();

async function doQuery<T extends { [key: string]: arrow.DataType }>(query: string): Promise<T[]> {
  const conn = await db.connect()
  await conn.send(`USE eval`);
  const result = (await conn.query<T>(query)).toArray();
  conn.close();
  return result;
}

export type PolicyEval = {
  policy_uri: string;
  eval_name: string;
  value: number;
  replay_url: string | null;
}

export async function getPolicyEvals(metric: string): Promise<PolicyEval[]> {
  const queryResult = await doQuery<any>(`
    WITH my_metric AS (SELECT * FROM episode_metrics WHERE metric = '${metric}') 
    SELECT
      e.policy_key || ':v' || e.policy_version AS policy_uri, 
      e.eval_name, 
      AVG(m.value) as value, 
      ANY_VALUE(e.replay_url) AS replay_url 
    FROM my_metric m 
    JOIN episode_info e 
    ON m.episode_id = e.episode_id 
    GROUP BY e.eval_name, e.policy_key, e.policy_version
  `);
  return queryResult;
}

export async function getMetrics(): Promise<string[]> {
  const queryResult = await doQuery<{ metric: arrow.Utf8 }>(`SELECT DISTINCT metric FROM episode_metrics ORDER BY metric`);
  return queryResult.map(row => row.metric.toString());
}