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
- [OK] `README.md` documentant le projet
- [OK] `.gitkeep` dans tous les dossiers

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

### 4. **Données**
- [OK] Structure `data/` prête avec sous-dossiers :
  - `data/raw/` (vidéos brutes)
  - `data/processed/` (vidéos prétraitées et clips)
  - `data/prepared/` (données préparées)
  - `data/manifests/` (CSV manifests)

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
- [ ] Modèle multimodal entraîné conjointement
- [ ] Modèle inspiré FGI (fine-grained inconsistency detection)
- [ ] Encodeurs pour audio et vidéo séparément

### 2. **Logique d'Entraînement** (`src/training/`)
- [OK] Boucle d'entraînement de base audio-only
- [OK] Gestion des checkpoints (save/load)
- [OK] Évaluation loss/accuracy hors entraînement
- [OK] Early stopping sur métrique de validation
- [ ] Learning rate scheduling
- [ ] Support GPU/Multi-GPU (si nécessaire)
- [ ] Logging d'expériences (TensorBoard, Weights&Biases, etc.)

### 3. **Métriques & Évaluation** (`src/evaluation/`)
- [OK] Accuracy, Precision, Recall, F1-score
- [OK] AUC-ROC
- [ ] AUC-PR
- [OK] Confusion matrix
- [OK] Courbes train loss/accuracy
- [OK] Courbes train vs val loss/accuracy
- [OK] Plot confusion matrix
- [ ] Courbes ROC et Précision-Recall
- [OK] Export `predictions/*.csv`
- [OK] Export `metrics/*.json`
- [ ] Rapport d'évaluation complet avec courbes
- [OK] Analyse d'accord et de désaccord entre modèles audio et vidéo

### 4. **Configuration** (`configs/`)
- [OK] `baseline_audio.yaml` - Configuration audio-only
- [OK] `baseline_video.yaml` - Configuration video-only
- [OK] `baseline_ensemble.yaml` - Configuration de fusion tardive
- [OK] `r3d18_video.yaml` - Baseline vidéo préentraînée Kinetics-400
- [OK] `fgi_preprocessing.yaml` - Contrat visuel préparatoire FGI
- [OK] `fgi_inspired.yaml` - Contrat multimodal strict, modèle marqué pending
- [ ] `multimodal.yaml` - Configuration d'un modèle entraîné conjointement
- [ ] `fgi_inspired.yaml` - Configuration FGI
- [OK] Loader Python pour lire et valider la config audio baseline
- [OK] Sélection indépendante de `video.preprocessing.name` et `model.name`

### 5. **Mise à jour main.py**
- [OK] Support de la commande `train`
- [OK] Support de la commande `eval`
- [OK] Chargement de configs YAML pour training audio-only
- [OK] Pipeline audio-only train/eval minimal
- [OK] Pipeline video-only train/eval minimal
- [OK] Commande `fgi-face-crops`
- [OK] Commande `fgi-data-smoke`
- [ ] Logging et rapports avancés

### 6. **Jobs cluster** (`jobs/`)
- [OK] Script Slurm audio baseline
- [OK] Script Slurm video baseline
- [OK] Paramétrage par variables d'environnement pour smoke tests
- [OK] Script d'évaluation d'un checkpoint
- [OK] Script d'évaluation ensemble
- [OK] Reprise via la variable `RESUME`

### 7. **Améliorations Futures**
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

**Modules implémentés** : 4/5 (80%)
- [OK] Data preprocessing & loading (100%)
- [EN COURS] Models (baselines unimodales terminées, modèle multimodal à faire)
- [OK] Training (audio/vidéo, checkpoints, reprise et early stopping)
- [OK] Evaluation (audio, vidéo, ensemble, CSV/JSON et plots)
- [OK] Configuration system (audio, vidéo et ensemble)

**Couverture** :
- Data layer : [OK] Complet avec tests
- Model layer : [EN COURS] Audio/vidéo testés, multimodal conjoint manquant
- Training layer : [OK] Runs cluster et reprise testés
- Eval layer : [OK] Audio, vidéo et ensemble testés

---

## Prochaines Étapes (Priorité Décroissante)

1. **URGENT** : Générer et contrôler les crops YuNet train/val/test
2. **IMPORTANT** : Implémenter les encodeurs audio et vidéo FGI
3. **IMPORTANT** : Ajouter les distances locales audio-visuelles et l'attention spatiale
4. **IMPORTANT** : Ajouter le trainer et l'évaluateur multimodaux
5. **IMPORTANT** : Ajouter les pseudo-fakes temporels au train uniquement
6. **IMPORTANT** : Ajouter balanced accuracy, F1 macro, AUC-PR et courbes ROC/PR
7. **IMPORTANT** : Comparer FGI aux baselines sur les mêmes splits

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

**Dernière mise à jour** : 12 juin 2026
**Statut** : Baselines/R3D-18 [OK] | Entrées FGI synchronisées [OK] | Modèle FGI [À FAIRE]
