from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import random

def split_dataset(
    source_dir,
    dest_dir,
    train_frac=0.7,
    val_frac=0.2,
    test_frac=0.1,
    seed=42,
    move_files=False,
    class_name_map=None
):
    """
    Divide un dataset organizado por carpetas de clase en train/val/test.
    - source_dir: carpeta que contiene subcarpetas por clase (ej: daisy, rose, ...)
    - dest_dir: carpeta destino que contendrá train/, val/, test/
    - class_name_map: dict opcional para renombrar carpetas, e.g. {'daisy':'margarita', ...}
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Las fracciones deben sumar 1.0"
    random.seed(seed)

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Extensiones de imagen a considerar
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif')

    # Mapa por defecto: inglés -> español (modifica si quieres otros nombres)
    default_map = {
        'daisy': 'margarita',
        'dandelion': 'diente_de_leon',
        'rose': 'rosa',
        'sunflower': 'girasol',
        'tulip': 'tulipan'
    }
    if class_name_map:
        # combina el mapa por defecto con el proporcionado (el usuario puede sobrescribir)
        mapping = {**default_map, **class_name_map}
    else:
        mapping = default_map

    # Itera por cada subcarpeta (clase)
    for cls in [d for d in source_dir.iterdir() if d.is_dir()]:
        # recoge archivos de imagen sin duplicados
        imgs = []
        for e in exts:
            imgs.extend(list(cls.glob(e)))
        # eliminar duplicados manteniendo orden
        imgs = list(dict.fromkeys(imgs))

        n = len(imgs)
        if n == 0:
            print(f"Aviso: la carpeta '{cls.name}' está vacía. Se omite.")
            continue

        # división: primero extraemos test, luego train/val desde lo que queda
        train_and_temp, test = train_test_split(imgs, test_size=test_frac, random_state=seed)
        # proporción relativa para val entre (train+val)
        rel_val = val_frac / (train_frac + val_frac)
        train, val = train_test_split(train_and_temp, test_size=rel_val, random_state=seed)

        splits = {'train': train, 'val': val, 'test': test}

        # nombre de la clase en español (si existe en mapping)
        spanish_name = mapping.get(cls.name, cls.name.replace(' ', '_').lower())

        for split_name, files in splits.items():
            target = dest_dir / split_name / spanish_name
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                dest_path = target / f.name
                if move_files:
                    shutil.move(str(f), str(dest_path))
                else:
                    shutil.copy2(str(f), str(dest_path))

        print(f"Clase '{cls.name}' ({n} archivos) -> train:{len(train)} val:{len(val)} test:{len(test)}")

    print("Split completado en:", dest_dir)

# Ejemplo de uso:
# split_dataset('ruta/a/flores_original', 'dataset', move_files=False)
# Si tus carpetas están exactamente como en la imagen: 'daisy','dandelion','rose','sunflower','tulip'
# entonces se crearán carpetas con: 'margarita','diente_de_leon','rosa','girasol','tulipan'
if __name__ == "__main__":
    # Carpeta original con las imágenes (ajusta si el nombre cambia)
    source = r"C:\personal\laynir\flowers"

    # Carpeta donde se guardará el nuevo dataset dividido
    destino = r"C:\personal\laynir\flores"

    # Ejecutar la separación
    split_dataset(
        source_dir=source,
        dest_dir=destino,
        move_files=False  # Cambia a True si quieres mover los archivos en lugar de copiarlos
    )
