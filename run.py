import argparse, json, logging, time, yaml, pandas as pd, numpy as np

def write_error(output, version, msg):
    data={"version":version,"status":"error","error_message":msg}
    with open(output,"w") as f: json.dump(data,f,indent=2)
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--log-file",required=True)
    a=p.parse_args()

    logging.basicConfig(filename=a.log_file, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    start=time.time()
    version="v1"

    try:
        logging.info("Job start")
        cfg=yaml.safe_load(open(a.config))
        for k in ["seed","window","version"]:
            if k not in cfg: raise ValueError(f"Missing config field: {k}")
        np.random.seed(cfg["seed"])
        version=cfg["version"]

        df=pd.read_csv(a.input)
        if df.empty: raise ValueError("Empty file")
        if "close" not in df.columns: raise ValueError("Missing required column: close")

        logging.info("Rows loaded: %s", len(df))
        rolling=df["close"].rolling(cfg["window"]).mean()
        signal=(df["close"]>rolling).astype(int)

        metrics={
            "version":version,
            "rows_processed":int(len(df)),
            "metric":"signal_rate",
            "value":round(float(signal.mean()),4),
            "latency_ms":int((time.time()-start)*1000),
            "seed":cfg["seed"],
            "status":"success"
        }
        with open(a.output,"w") as f: json.dump(metrics,f,indent=2)
        print(json.dumps(metrics))
    except Exception as e:
        err=write_error(a.output, version, str(e))
        print(json.dumps(err))
        raise

if __name__=="__main__":
    main()
