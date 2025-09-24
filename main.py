from pathlib import Path
import pandas as pd

from processamento_dados import data_processing


def main():
    df_path = Path("./dados/br-capes-filtrado.csv")

    if not df_path.exists():
        print("Gerando o arquivo")
        data_processing()
    df = pd.read_csv(df_path)


if __name__ == "__main__":
    main()
