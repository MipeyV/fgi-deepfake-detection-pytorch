# SUIVI du Projet FGI Deepfake Detection

## Contexte & Objectifs

### Repo Original
- **Repo inspirant** : [aseuteurideu/FGI](https://github.com/aseuteurideu/FGI)
- **Papier** : *Detecting Audio-Visual Deepfakes with Fine-Grained Inconsistencies* (BMVC 2024)
- **Approche originale** : Détection de deepfakes via détection d'inconsistances fine-grained entre audio et vidéo

### Objectif du Projet
Ce projet **ne réutilise pas directement le code original**, mais vise à **réimplémenter les idées principales** pour :
- [OK] Meilleure compréhension de la méthode FGI
- [OK] Code modulaire et maintenable
- [OK] Pipeline PyTorch propre
- [OK] Possibilité d'expérimentation et d'amélioration

### Approche Adoptée
1. Commencer par un baseline simple (audio-only ou video-only)
2. Construire un pipeline multimodal
3. Implémenter une approche inspirée par FGI
4. Exécuter des expériences et comparer les résultats

---

## ÉTAT ACTUEL - Ce qui est COMPLÉTÉ

### 1. **Infrastructure & Structure**
- [OK] Projet structuré avec modules clairs (`src/data`, `src/models`, `src/training`, `src/evaluation`)
- [OK] `requirements.txt` avec dépendances PyTorch + OpenCV + Pandas
- [OK] `pytest.ini` pour tests unitaires
- [OK] `pyproject.toml` avec configuration Ruff lint + format
- [OK] GitHub Actions CI avec lint, format, compileall et tests
- [OK] `README.md` documentant le projet
- [OK] `.gitkeep` dans tous les dossiers
- [OK] Organisation des modèles en sous-packages (`baselines/audio`, `baselines/video`, `fgi`)

### 2. **Pipeline de Prétraitement** (`src/data/`)
- [OK] **preprocessing_pipeline.py** :
  - Découverte de vidéos (réelles et fakes)
  - Normalisation des vidéos avec ffmpeg
  - Extraction de frames avec ffmpeg
  - Extraction d'audio avec ffmpeg
  - Création de clips synchronisés (audio-visual)
  - Écriture de manifests CSV

- [OK] **dataset.py** :
  - Classe `DeepFakeClipDataset` (PyTorch Dataset)
  - Chargement de frames et audio
  - Support des transformations de frames
  - Validation des manifests
  - Encodage des labels (real=0, fake=1)

- [OK] **dataloader.py** :
  - Fonction `create_dataloader()` pour créer des dataloaders
  - Fonction `collate_deepfake_batch()` pour collation personnalisée
  - Support batch_size, shuffle, num_workers configurables

- [OK] **Pipeline vidéo configurable** :
  - `resize_square`
  - `resize_center_crop`
  - `resize_normalize`
  - normalisation FGI configurable vers `[-1, 1]`
  - sélection indépendante du modèle via YAML

- [OK] **Préprocessing facial FGI hors ligne** :
  - détecteur YuNet via OpenCV et modèle ONNX explicite
  - fallback Haar disponible pour le développement
  - association temporelle des visages par IoU
  - sélection de la piste faciale la plus longue
  - boîte carrée stabilisée sur toutes les frames du clip
  - marge et taille de sortie configurables
  - copie de l'audio synchronisé
  - conservation des colonnes de split dans les manifests FGI
  - seuil minimal de frames avec visage détecté
  - politique `error` ou `skip` pour les clips sans visage fiable
  - planche contact PNG début/milieu/fin pour contrôle qualité

- [OK] **Dataset multimodal FGI strict** :
  - chargement conjoint de 30 crops faciaux et de l'audio synchronisé
  - vidéo `[batch, 30, 3, 224, 224]`
  - audio mono `[batch, 48000]`
  - validation stricte du nombre de frames
  - validation PCM 16 bits, 48 kHz et 48 000 échantillons
  - normalisation vidéo et audio vers `[-1, 1]`
  - collate multimodal dédié
  - commande smoke sans modèle

- [OK] **split_manifest.py** :
  - Hash-based split des données (train/val/test = 7:2:1)
  - Validation de manifests
  - Utilise SHA256 pour reproducibilité

- [OK] **prepare_dfdc.py** :
  - Préparation du dataset DFDC
  - Extraction des métadonnées
  - Organisation des données

### 3. **Tests** (`tests/`)
- [OK] Fixtures de test dans `tests/data/helpers.py`
- [OK] Tests unitaires :
  - `test_dataset.py`
  - `test_dataloader.py`
  - `test_preprocessing_pipeline.py`
  - `test_split_manifest.py`
- [OK] CI `Lint, format, and test` :
  - `ruff check .`
  - `ruff format --check .`
  - `python -m compileall -q main.py src tests`
  - `pytest`

### 4. **Données**
- [OK] Structure `data/` prête avec sous-dossiers :
  - `data/raw/` (vidéos brutes)
  - `data/processed/` (vidéos prétraitées et clips)
  - `data/prepared/` (données préparées)
  - `data/manifests/` (CSV manifests)
- [OK] Documentation du flux Kaggle/DFDC :
  - inscription et acceptation de la compétition Kaggle
  - vérification d'identité Kaggle si nécessaire
  - installation de FFmpeg avant preprocessing
  - préparation `real/` / `fake/`
  - preprocessing initial et création des manifests
- [OK] Analyse statique enrichie du dataset :
  - counts clips et vidéos par split/classe
  - distribution des clips par vidéo
  - estimation des durées vidéo depuis les clips prétraités
  - confirmation que le subset courant a `10` clips par vidéo, soit environ `10s`
  - checks leakage, doublons, labels et chemins manquants

### 5. **Configuration initiale** (`configs/`)
- [OK] **baseline_audio.yaml** :
  - Contrat de configuration pour la première baseline audio-only
  - Chemins des manifests train/val/test
  - Paramètres audio et mel-spectrogram
  - Hyperparamètres modèle, entraînement, validation, évaluation
  - Dossiers de checkpoints et logs

### 6. **Features audio** (`src/data/`)
- [OK] **audio_feature.py** :
  - Conversion waveform -> mel-spectrogram avec `torch.stft`
  - Banque de filtres mel triangulaires sans dépendance externe
  - Support entrée simple `[channels, samples]`
  - Support batch `[batch_size, channels, samples]`
  - Builder compatible avec la configuration YAML

### 7. **Chargement de configuration** (`src/config.py`)
- [OK] Lecture YAML via `load_config()`
- [OK] Validation de la configuration baseline audio
- [OK] Vérification des sections obligatoires
- [OK] Vérification des paramètres numériques clés
- [OK] Tests sur le vrai fichier `configs/baseline_audio.yaml`

### 8. **Training audio-only** (`src/training/`)
- [OK] **trainer.py** :
  - Résolution du device (`auto`, `cpu`, `cuda`)
  - Construction d'optimizer (`adam`, `sgd`)
  - Entraînement d'une epoch audio-only
  - Évaluation loss/accuracy sans mise à jour des poids
  - Support `max_batches` pour smoke tests et overfit rapide
  - Test réel sur mini-batch depuis `data/manifests/train_manifest.csv`
- [OK] **checkpoints.py** :
  - Sauvegarde PyTorch `.pt`
  - `last.pt` pour la dernière epoch
  - `best.pt` selon la métrique configurée
  - Chargement des poids pour évaluation

### 9. **Organisation des runs** (`src/runs.py`)
- [OK] Génération de `run_id`
- [OK] Création de dossiers `runs/<experiment>/<run_id>/`
- [OK] Sous-dossiers :
  - `checkpoints/`
  - `logs/`
  - `metrics/`
  - `predictions/`
  - `plots/`
- [OK] Copie de la config utilisée dans `config.yaml`
- [OK] Sauvegarde de la commande dans `command.txt`
- [OK] Sauvegarde de l'état Git dans `git.json`
- [OK] Sauvegarde des métadonnées dans `metadata.json`

### 10. **Entrée CLI train** (`main.py`)
- [OK] Commande `python3 main.py train --config ...`
- [OK] Support `--epochs`
- [OK] Support `--max-batches`
- [OK] Support `--batch-size`
- [OK] Support `--run-id`
- [OK] Support `--runs-root`
- [OK] Support `--device`
- [OK] Mode audio-only sans chargement des frames vidéo
- [OK] Écriture de `metrics/train_metrics.json`
- [OK] Écriture de `checkpoints/last.pt`
- [OK] Écriture de `checkpoints/best.pt`
- [OK] Validation par epoch si `val_manifest` existe
- [OK] Reprise complète avec `--resume` depuis `last.pt`
- [OK] Sauvegarde progressive de l'historique après chaque epoch
- [OK] Évaluation test automatique du meilleur checkpoint en fin de training

### 11. **Évaluation audio-only** (`src/evaluation/`)
- [OK] **metrics.py** :
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC-ROC binaire
  - Confusion matrix (`tn`, `fp`, `fn`, `tp`)
- [OK] **evaluator.py** :
  - Boucle d'inférence audio-only
  - Collecte des probabilités `real/fake`
  - Export CSV de prédictions par clip
  - Export JSON des métriques globales
- [OK] Commande `python3 main.py eval --config ...`
- [OK] Chargement optionnel d'un checkpoint entraîné via `--checkpoint`

### 12. **Plots de training** (`src/evaluation/plots.py`)
- [OK] Lecture de `metrics/train_metrics.json`
- [OK] Génération de `plots/training_history.svg`
- [OK] Génération de `plots/train_loss.svg`
- [OK] Génération de `plots/train_accuracy.svg`
- [OK] Courbe loss par epoch
- [OK] Courbe accuracy par epoch
- [OK] Courbes train vs val pour loss/accuracy
- [OK] Génération automatique après `main.py train`
- [OK] Génération de `plots/<split>_confusion_matrix.svg`
- [OK] Génération automatique après `main.py eval`
- [OK] Axes gradués et grille sur les courbes SVG

### 13. **Baseline vidéo et exécution cluster**
- [OK] Baseline vidéo 3D CNN
- [OK] Training et évaluation vidéo
- [OK] Scripts Slurm audio et vidéo sans buffering stdout
- [OK] Job d'évaluation indépendant depuis un checkpoint
- [OK] Reprise d'un entraînement interrompu dans un nouveau job

### 14. **Comparaison audio-vidéo et ensemble**
- [OK] Commande `main.py ensemble-eval`
- [OK] Fusion tardive par moyenne des probabilités audio et vidéo
- [OK] Taux d'accord et nombre de désaccords
- [OK] Export CSV détaillé par clip
- [OK] Métriques séparées audio, vidéo et ensemble
- [OK] Job Slurm `jobs/eval_ensemble.sbatch`
- [EN COURS] Job ensemble `843121` en attente de ressources

---

## À FAIRE - Ce qui est MANQUANT

### 1. **Modèles** (`src/models/`)
- [OK] Baseline audio-only (`src/models/audio_models.py`)
- [OK] Baseline video-only (`src/models/video_models.py`)
- [EN COURS] Fusion tardive audio-vidéo disponible pour évaluation
- [OK] Modèle multimodal entraîné conjointement
- [OK] Modèle inspiré FGI (fine-grained inconsistency detection)
- [OK] Encodeurs FGI audio et vidéo séparés
- [OK] Projection commune `[B, 128, 15]`
- [OK] Features vidéo locales `[B, 128, 15, 28, 28]`
- [OK] Distance locale et attention FGI

### 2. **Logique d'Entraînement** (`src/training/`)
- [OK] Boucle d'entraînement de base audio-only
- [OK] Gestion des checkpoints (save/load)
- [OK] Évaluation loss/accuracy hors entraînement
- [OK] Early stopping sur métrique de validation
- [OK] Boucles train/validation FGI multimodales
- [OK] Évaluation FGI avec métriques et prédictions CSV
- [ ] Learning rate scheduling
- [ ] Support GPU/Multi-GPU (si nécessaire)
- [ ] Logging d'expériences (TensorBoard, Weights&Biases, etc.)

### 3. **Métriques & Évaluation** (`src/evaluation/`)
- [OK] Accuracy, Precision, Recall, F1-score
- [OK] Balanced accuracy, spécificité et F1 macro
- [OK] AUC-ROC
- [OK] AUC-PR via average precision
- [OK] Confusion matrix
- [OK] Courbes train loss/accuracy
- [OK] Courbes train vs val loss/accuracy
- [OK] Plot confusion matrix
- [ ] Courbes ROC et Précision-Recall
- [OK] Export `predictions/*.csv`
- [OK] Export `metrics/*.json`
- [OK] Calibration du seuil sur validation
- [OK] Agrégation des probabilités et métriques par vidéo
- [ ] Rapport d'évaluation complet avec courbes
- [OK] Analyse d'accord et de désaccord entre modèles audio et vidéo

### 4. **Configuration** (`configs/`)
- [OK] `baseline_audio.yaml` - Configuration audio-only
- [OK] `baseline_video.yaml` - Configuration video-only
- [OK] `baseline_ensemble.yaml` - Configuration de fusion tardive
- [OK] `r3d18_video.yaml` - Baseline vidéo préentraînée Kinetics-400
- [OK] `fgi_preprocessing.yaml` - Contrat visuel préparatoire FGI
- [OK] `fgi_inspired.yaml` - Configuration FGI train/eval
- [OK] Loader Python pour lire et valider la config audio baseline
- [OK] Sélection indépendante de `video.preprocessing.name` et `model.name`
- [OK] Configuration Ruff centralisée dans `pyproject.toml`

### 5. **Mise à jour main.py**
- [OK] Support de la commande `train`
- [OK] Support de la commande `eval`
- [OK] Chargement de configs YAML pour training audio-only
- [OK] Pipeline audio-only train/eval minimal
- [OK] Pipeline video-only train/eval minimal
- [OK] Commande `fgi-face-crops`
- [OK] Commande `fgi-data-smoke`
- [OK] Commande `fgi-encoder-smoke`
- [OK] Commande `fgi-model-smoke`
- [OK] Support FGI dans les commandes `train` et `eval`
- [ ] Logging et rapports avancés

### 6. **Jobs cluster** (`jobs/`)
- [OK] Script Slurm audio baseline
- [OK] Script Slurm video baseline
- [OK] Script Slurm entraînement FGI
- [OK] Paramétrage par variables d'environnement pour smoke tests
- [OK] Script d'évaluation d'un checkpoint
- [OK] Script d'évaluation ensemble
- [OK] Reprise via la variable `RESUME`

### 7. **CI, qualité et documentation**
- [OK] Workflow GitHub Actions `.github/workflows/ci.yml`
- [OK] Installation CI de FFmpeg et `libgl1` pour les tests OpenCV/vidéo
- [OK] Ruff linting activé (`E`, `F`, `I`, `UP`, `B`)
- [OK] Ruff formatting vérifié en CI
- [OK] Compilation Python vérifiée avant tests
- [OK] README mis à jour avec :
  - setup FFmpeg
  - téléchargement Kaggle/DFDC
  - préparation des données
  - preprocessing complet
  - split manifests
  - note sur les vidéos uniformes de 10 secondes dans le subset courant

### 8. **Améliorations Futures**
- [ ] Optimisation des hyperparamètres
- [ ] Data augmentation cohérente au niveau du clip
- [ ] Stratégies de régularisation
- [OK] Première technique d'ensemble par moyenne des probabilités
- [EN COURS] Analyse des erreurs et désaccords par clip
- [ ] Documentation d'expériences (notebooks/)

---

## Approche Comparative

### Différences vs Repo Original (FGI)
| Aspect | FGI Original | Ce Projet |
|--------|-------------|-----------|
| **Code base** | Implémentation complète du papier | Réimplémentation modulaire |
| **Modularité** | Peut être monolithique | Modules séparés et testables |
| **Tests** | Peut manquer de couverture | Tests unitaires dès le départ |
| **Configurabilité** | Codes en dur (hardcoded) | Pipelines et modèles sélectionnés en YAML |
| **Documentation** | Peut être minimale | README, journal et suivi maintenus |
| **Maintenabilité** | Optimisée pour recherche | Optimisée pour réutilisabilité |

### Améliorations Apportées
- [OK] Pipeline de prétraitement entièrement fonctionnel
- [OK] Structure PyTorch propre (Dataset, DataLoader)
- [OK] Tests unitaires pour chaque composant
- [OK] Manifest-based dataset management (reproducible)
- [OK] Support multi-format vidéo via ffmpeg
- [OK] Hash-based split reproducible

---

## Métriques de Progression

**Modules implémentés** : 5/5 (100%)
- [OK] Data preprocessing & loading (100%)
- [OK] Models (baselines unimodales, R3D-18 et FGI multimodal)
- [OK] Training (audio/vidéo, checkpoints, reprise et early stopping)
- [OK] Evaluation (audio, vidéo, ensemble, CSV/JSON et plots)
- [OK] Configuration system (audio, vidéo et ensemble)

**Couverture** :
- Data layer : [OK] Complet avec tests
- Model layer : [OK] Audio, vidéo et FGI multimodal testés
- Training layer : [OK] Runs cluster et reprise testés
- Eval layer : [OK] Audio, vidéo et ensemble testés

---

## Prochaines Étapes (Priorité Décroissante)

1. **URGENT** : Contrôler les contact sheets YuNet et le nombre de clips skip
2. **IMPORTANT** : Comparer FGI aux baselines sur les mêmes splits et niveaux
   clip/vidéo
3. **IMPORTANT** : Ajouter les pseudo-fakes temporels au train uniquement
4. **IMPORTANT** : Ajouter les courbes ROC et précision-rappel
5. **IMPORTANT** : Documenter les résultats expérimentaux finaux dans le README
   ou un notebook dédié

---

## Mises à jour depuis le 13 juin 2026

### Qualité, CI et formatage
- [OK] Ajout de Ruff dans `pyproject.toml` pour lint et format.
- [OK] Ajout d'un workflow GitHub Actions `CI`.
- [OK] Le job CI exécute désormais :
  - installation de FFmpeg et `libgl1`
  - installation des dépendances Python
  - `ruff check .`
  - `ruff format --check .`
  - `python -m compileall -q main.py src tests`
  - `pytest`
- [OK] La protection de branche attend maintenant le check `Lint, format, and test`.

### Réorganisation du code modèle
- [OK] Refactor des modèles en packages plus explicites :
  - `src/models/baselines/audio/`
  - `src/models/baselines/video/`
  - `src/models/fgi/`
- [OK] Séparation des factories, validations et implémentations vidéo.
- [OK] Imports adaptés dans les configs, tests et points d'entrée.

### Documentation dataset
- [OK] README enrichi avec le chemin complet de préparation des données :
  - rejoindre la compétition Kaggle DFDC
  - vérifier son identité Kaggle si nécessaire
  - télécharger/extracter les données
  - préparer `data/prepared/dfdc/real` et `data/prepared/dfdc/fake`
  - lancer `main.py preprocess`
  - créer les manifests train/val/test
- [OK] README enrichi avec l'installation de FFmpeg.
- [OK] README précise que le preprocessing par défaut crée des clips de `1s`
  et que le subset actuel produit `10` clips par vidéo, donc environ `10s`.

### Analyse dataset
- [OK] Notebook `notebooks/dataset_static_analysis.ipynb` mis à jour.
- [OK] Ajout des durées vidéo estimées depuis le nombre de clips.
- [OK] Ajout du range de durée, des durées uniques et du flag d'uniformité.
- [OK] Confirmation dans l'analyse que les vidéos du subset courant sont
  uniformes : `10` clips par vidéo, soit environ `10s`.
- [OK] Le résumé compact affiche maintenant la durée moyenne estimée par classe
  `real/fake` et le range global.

---

## Notes Importantes

- **FFmpeg requis** : Le projet utilise ffmpeg pour traiter les vidéos
- **Pas de données incluses** : Les données sont gérées via manifests CSV (`.gitignore`)
- **Reproducibilité** : Le split train/val/test utilise SHA256(video_id) % 10 pour garantir la reproductibilité
- **Format données** : Chaque clip = vidéo + audio synchronisés au même fps/sample_rate

---

## Références Utiles

- [FGI Paper (BMVC 2024)](https://arxiv.org/abs/2408.06753)
- [Repo Original](https://github.com/aseuteurideu/FGI)
- [PyTorch Dataset & DataLoader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [DFDC Dataset](https://deepfakedetectionchallenge.ai/) - Si utilisé

---

**Dernière mise à jour** : 25 juin 2026
**Statut** : Modèle FGI [OK] | Calibration [OK] | CI/Ruff [OK] | Analyse dataset [OK] | Comparaison [À FAIRE]
