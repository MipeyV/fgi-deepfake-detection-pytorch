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

---

## À FAIRE - Ce qui est MANQUANT

### 1. **Modèles** (`src/models/`)
- [OK] Baseline audio-only (`src/models/audio_models.py`)
- [ ] Baseline video-only (ex: 3D CNN ou ViT)
- [ ] Modèle multimodal (fusion audio-vidéo)
- [ ] Modèle inspiré FGI (fine-grained inconsistency detection)
- [ ] Encodeurs pour audio et vidéo séparément

### 2. **Logique d'Entraînement** (`src/training/`)
- [OK] Boucle d'entraînement de base audio-only
- [ ] Gestion des checkpoints (save/load)
- [OK] Évaluation loss/accuracy hors entraînement
- [ ] Learning rate scheduling
- [ ] Support GPU/Multi-GPU (si nécessaire)
- [ ] Logging d'expériences (TensorBoard, Weights&Biases, etc.)

### 3. **Métriques & Évaluation** (`src/evaluation/`)
- [ ] Accuracy, Precision, Recall, F1-score
- [ ] AUC-ROC et AUC-PR
- [ ] Confusion matrix
- [ ] Courbes ROC et Précision-Recall
- [ ] Rapport d'évaluation complet

### 4. **Configuration** (`configs/`)
- [OK] `baseline_audio.yaml` - Configuration audio-only
- [ ] `baseline_video.yaml` - Configuration video-only
- [ ] `multimodal.yaml` - Configuration multimodal
- [ ] `fgi_inspired.yaml` - Configuration FGI
- [OK] Loader Python pour lire et valider la config audio baseline
- [ ] Format YAML standardisé pour les futures configs

### 5. **Mise à jour main.py**
- [ ] Support de commandes supplémentaires (train, eval)
- [ ] Chargement de configs YAML
- [ ] Pipeline complet d'entraînement et d'évaluation
- [ ] Logging et rapports

### 6. **Améliorations Futures**
- [ ] Optimisation des hyperparamètres
- [ ] Data augmentation pour audio et vidéo
- [ ] Stratégies de régularisation
- [ ] Techniques d'ensemble
- [ ] Analyse des erreurs et visualisations
- [ ] Documentation d'expériences (notebooks/)

---

## Approche Comparative

### Différences vs Repo Original (FGI)
| Aspect | FGI Original | Ce Projet |
|--------|-------------|-----------|
| **Code base** | Implémentation complète du papier | Réimplémentation modulaire |
| **Modularité** | Peut être monolithique | Modules séparés et testables |
| **Tests** | Peut manquer de couverture | Tests unitaires dès le départ |
| **Configurabilité** | Codes en dur (hardcoded) | YAML configs (future) |
| **Documentation** | Peut être minimale | Documentation complète (future) |
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

**Modules implémentés** : 1/5 (20%)
- [OK] Data preprocessing & loading (100%)
- [EN COURS] Models (baseline audio-only implémentée)
- [EN COURS] Training (epoch audio-only + évaluation simple)
- [TODO] Evaluation (0%)
- [EN COURS] Configuration system (baseline audio + loader validés)

**Couverture** :
- Data layer : [OK] Complet avec tests
- Model layer : [EN COURS] Baseline audio-only testée
- Training layer : [EN COURS] Mini-train audio-only testé
- Eval layer : [TODO] Non started

---

## Prochaines Étapes (Priorité Décroissante)

1. **URGENT** : Créer la branche `feature/training-model`
2. **URGENT** : Standardiser les sorties de run (`runs/`, run_id, copies config)
3. **IMPORTANT** : Ajouter checkpoints save/load
4. **IMPORTANT** : Implémenter les métriques d'évaluation
5. **IMPORTANT** : Produire les prédictions CSV sur le test set
6. **NICE-TO-HAVE** : Ajouter logging et visualization
7. **NICE-TO-HAVE** : Optimisation hyperparamètres

---

## Notes Importantes

- **FFmpeg requis** : Le projet utilise ffmpeg pour traiter les vidéos
- **Pas de données incluses** : Les données sont gérées via manifests CSV (`.gitignore`)
- **Reproducibilité** : Le split train/val/test utilise SHA256(video_id) % 10 pour garantir la reproductibilité
- **Format données** : Chaque clip = vidéo + audio synchronisés au même fps/sample_rate

---

## Références Utiles

- [FGI Paper (BMVC 2024)](https://arxiv.org/abs/xxxx.xxxxx) - *À confirmer*
- [Repo Original](https://github.com/aseuteurideu/FGI)
- [PyTorch Dataset & DataLoader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [DFDC Dataset](https://deepfakedetectionchallenge.ai/) - Si utilisé

---

**Dernière mise à jour** : Juin 2, 2026  
**Statut** : Phase de prétraitement terminée [OK] | Phase modélisation en attente [EN COURS]
