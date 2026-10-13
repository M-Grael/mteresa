import cmdstanpy

print("Iniciando instalação (isso pode demorar de 2 a 10 minutos)...")

cmdstanpy.install_cmdstan(
    compiler=True, 
    dir=r"D:\03_renovacao\ASU\cmdstan", 
    overwrite=True
)

print("Instalação concluída com sucesso!")
print("A pasta correta para usar no seu test.py é:")
print(cmdstanpy.cmdstan_path())