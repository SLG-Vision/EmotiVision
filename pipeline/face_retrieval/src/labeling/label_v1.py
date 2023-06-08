import os
import json
from PIL import Image

def main(folder_name):
    # Controllo se la cartella esiste
    if not os.path.exists(folder_name):
        print(f"La cartella '{folder_name}' non esiste.")
        return

    # Controllo che tutti i file nella cartella siano immagini
    for filename in os.listdir(folder_name):
        if not is_image(os.path.join(folder_name, filename)):
            print(f"Il file '{filename}' non è un'immagine.")
            return

    # Chiedo all'utente se rinominare in modo progressivo o lasciare i nomi originali
    while True:
        choice = input("Vuoi rinominare le immagini in modo progressivo? (y/n):    ")
        if choice in ['y', 'n']:
            break

    if choice == 'y':
        # Rinomino le immagini in ordine numerico crescente
        images = sorted([f for f in os.listdir(folder_name) if is_image(os.path.join(folder_name, f))], key=lambda f: int(os.path.splitext(f)[0]))
        for i, image_name in enumerate(images):
            old_name = os.path.join(folder_name, image_name)
            new_name = os.path.join(folder_name, f"{i}.jpg")
            os.rename(old_name, new_name)
    else:
        print("Lascio i nomi originali.")

    # Chiedo all'utente di etichettare le immagini
    labels = {}
    images = sorted([f for f in os.listdir(folder_name) if is_image(os.path.join(folder_name, f))], key=lambda f: int(os.path.splitext(f)[0]))
    try:
        for image_name in images:
            image_path = os.path.join(folder_name, image_name)
            image = Image.open(image_path)
            image.show()
            while True:
                label = input(f"Inserire label per l'immagine '{image_name}' (y/n)")
                if label in ['y', 'n']:
                    labels[image_path] = label
                    break
    except KeyboardInterrupt:
        print("\nInterruzione del programma. Salvataggio del progresso...")
    else:
        print("\nEtichettazione completata con successo.")

    # Salvo il dizionario JSON
    with open('labels.json', 'w') as f:
        json.dump(labels, f)
        print("Salvataggio del file JSON completato.")

def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

if __name__ == '__main__':
    folder_name = input("Inserisci il nome della cartella: ")
    main(folder_name)
