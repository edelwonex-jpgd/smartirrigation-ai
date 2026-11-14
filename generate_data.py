from model import DATASET, make_synthetic_dataset

if __name__ == '__main__':
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    df=make_synthetic_dataset()
    df.to_csv(DATASET,index=False)
    print(f'Dibuat {len(df)} baris data simulasi: {DATASET}')
