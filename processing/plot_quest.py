import matplotlib.pyplot as plt

def plotar_vetores_quest(vetor1, vetor2, titulo="Gráfico de Vetores", label1="Vetor 1", label2="Vetor 2"):
   
    plt.figure(figsize=(10, 6))  # Define o tamanho da figura
    
    # Plota o primeiro vetor como pontos verdes
    plt.plot(vetor1, 'go', label=label1, markersize=8)  # 'go' significa green (verde), circles (o)
    
    # Plota o segundo vetor como pontos azuis
    plt.plot(vetor2, 'ro', label=label2, markersize=8)  # 'bo' significa blue (azul), circles (o)
    
    plt.title(titulo)
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.savefig("vector_comparison.png")
