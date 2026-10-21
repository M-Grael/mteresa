import os

# Nomes das pastas que devem ser ignoradas
pastas_ignoradas = {'.git', 'venv', '.venv', 'sandbox'}

for raiz, pastas, arquivos in os.walk('.'):
    # Modifica a lista 'pastas' em tempo de execução para não entrar nelas
    pastas[:] = [p for p in pastas if p not in pastas_ignoradas]
    
    for arquivo in arquivos:
        print(os.path.join(raiz, arquivo))