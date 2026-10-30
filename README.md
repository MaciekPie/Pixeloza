# ![Icon](data/mascot.png) Pixeloza

Generator pixel-artowych postaci - projekt porównujący różne architektury generatywne na tym samym zbiorze danych. Aplikacja wykorzystuje interfejs Gradio do generowania obrazów na podstawie promptów tekstowych oraz porównywania wyników poszczególnych modeli.


## Cel projektu

Celem projektu jest zbadanie i porównanie możliwości różnych architektur generatywnych w zadaniu tworzenia postaci w stylu pixel art. Modele są trenowane na tym samym zbiorze danych, co pozwala na analizę różnic w jakości, stylu i charakterze generowanych obrazów.


## Zbiór danych

W projekcie wykorzystano zbiór danych:
[`jiovine/pixel-art-nouns`](https://huggingface.co/datasets/jiovine/pixel-art-nouns)


## Architektury

W projekcie analizowane są następujące architektury:
- VQ-VAE,
- VQ-GAN,
- Dyfuzja.


## Interfejs aplikacji

Aplikacja wykorzystuje bibliotekę Gradio do udostępnienia interfejsu umożliwiającego:
- wprowadzanie promptów tekstowych,
- generowanie postaci pixel-art,
- porównywanie rezultatów różnych modeli.


## Technologie
- Python,
- Gradio.


## Autorzy

| Autor | Wkład w projekt |
|---|---|
| [@jakubbaldyga](https://github.com/jakubbaldyga) | Transformery |
| [@MartaStankiewicz](https://github.com/MartaStankiewicz) | Dyfuzja |
| [@Patrycjadz0](https://github.com/Patrycjadz0) | Dyfuzja |
| [@Glupotka](https://github.com/Glupotka) | VQ-VAE |
| [@MaciekPie](https://github.com/MaciekPie) | VQ-GAN |
