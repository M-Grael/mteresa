import mteresa as mts
import os

caminho_do_repositorio = r'D:\03_renovacao\ASU\graja\repositorio' 

for pasta_atual, subpastas, arquivos in os.walk(caminho_do_repositorio):
    print(f"Pasta: {pasta_atual}")
    for arquivo in arquivos:
        print(f"  - {arquivo}")
