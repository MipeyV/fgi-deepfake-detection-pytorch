# JOURNAL des réalisations

Ce journal retrace les réalisations du projet par périodes.  
Contrairement à `SUIVI.md`, qui sert d'état global d'avancement, ce fichier raconte ce qui a été fait, pourquoi, et dans quelle logique.

---

## Mars 2026 - Initialisation du projet

### Réalisations
- Création de la structure initiale du dépôt.
- Mise en place des dossiers principaux :
  - `src/`
  - `configs/`
  - `tests/`
  - `experiments/`
  - `jobs/`
  - `notebooks/`
- Ajout des premiers fichiers de base :
  - `README.md`
  - `.gitignore`
  - `requirements.txt`

### Logique
L'objectif était de partir sur une base propre et modulaire, plutôt que de recopier directement le dépôt original FGI.  
Le projet a été organisé pour séparer clairement les responsabilités : données, modèles, entraînement, évaluation et configurations.

### Résultat
Le dépôt dispose d'une architecture lisible, adaptée à une réimplémentation progressive en PyTorch.

---

## Mai 2026 - Pipeline de prétraitement

### Réalisations
- Implémentation du pipeline de prétraitement dans `src/data/preprocessing_pipeline.py`.
- Ajout d'une commande CLI dans `main.py` pour lancer le preprocessing.
- Gestion des vidéos réelles et fake.
- Normalisation vidéo via ffmpeg.
- Extraction des frames.
- Extraction de l'audio.
- Création de clips audio-visuels synchronisés.
- Écriture d'un manifest CSV décrivant les clips générés.

### Logique
Avant de travailler sur les modèles, il fallait fiabiliser l'entrée du pipeline.  
Le choix a été de transformer les vidéos brutes en clips standardisés, chacun contenant :
- des frames image,
- un fichier `audio.wav`,
- des métadonnées dans un manifest.

Cette étape rend les modèles indépendants des formats vidéo bruts et permet de travailler avec des données reproductibles.

### Résultat
Le projet peut préparer un dataset exploitable à partir de vidéos source.

---

## Mai 2026 - Tests du preprocessing

### Réalisations
- Ajout de tests unitaires pour le pipeline de prétraitement.
- Mise en place de `pytest.ini`.
- Ajout de fixtures et helpers dans `tests/data/helpers.py`.

### Logique
Le preprocessing touche aux fichiers, aux chemins, aux manifests et à ffmpeg.  
C'est une partie fragile du projet, donc elle devait être testée tôt pour éviter de construire les modèles sur une base instable.

### Résultat
Les composants de preprocessing sont couverts par des tests, ce qui facilite les modifications futures.

---

## Mai 2026 - Dataset PyTorch et DataLoader

### Réalisations
- Implémentation de `DeepFakeClipDataset` dans `src/data/dataset.py`.
- Chargement des frames depuis les dossiers de clips.
- Chargement de l'audio depuis `audio.wav`.
- Encodage des labels :
  - `real -> 0`
  - `fake -> 1`
- Validation des manifests.
- Implémentation de `create_dataloader()` dans `src/data/dataloader.py`.
- Ajout de `collate_deepfake_batch()` pour former des batches PyTorch.

### Logique
Une fois les clips prétraités, il fallait une interface standard PyTorch.  
Le dataset lit les manifests et renvoie des dictionnaires contenant les tenseurs, les labels et les métadonnées.

Le DataLoader garde une séparation claire :
- le Dataset sait lire un exemple,
- le collate sait assembler un batch,
- le code d'entraînement pourra ensuite consommer directement ces batches.

### Résultat
La couche data est prête pour l'entraînement PyTorch.

---

## Mai 2026 - Split reproductible des manifests

### Réalisations
- Implémentation de `src/data/split_manifest.py`.
- Split train/validation/test basé sur le hash du `video_id`.
- Ratio logique :
  - train : 70 %
  - validation : 20 %
  - test : 10 %
- Ajout de tests dédiés.

### Logique
Le split doit être reproductible et éviter les fuites entre splits.  
Le choix d'un hash par `video_id` permet de garder tous les clips d'une même vidéo dans le même split.

### Résultat
Le projet peut générer des manifests train/val/test stables et cohérents.

---

## Juin 2026 - Stabilisation pour environnement cluster

### Réalisations
- Corrections de bugs pour rendre le code compatible avec l'exécution sur cluster.
- Nettoyage et stabilisation de la base existante.
- Mise à jour du suivi global du projet.

### Logique
Avant de passer à l'entraînement, le code devait être suffisamment robuste pour tourner hors environnement local.  
La priorité était donc de corriger les problèmes bloquants et de consolider l'existant.

### Résultat
La base data/preprocessing est considérée comme terminée pour une première version.

---

## Juin 2026 - Démarrage de la phase training/model

### Réalisations
- Création de la branche `feature/training-model`.
- Création de `configs/baseline_audio.yaml`.
- Définition d'un contrat de configuration pour une première baseline audio-only :
  - manifests train/val/test,
  - paramètres audio,
  - paramètres mel-spectrogram,
  - architecture du modèle,
  - hyperparamètres d'entraînement,
  - validation, métriques, checkpoints et logs.
- Mise à jour de `SUIVI.md` pour distinguer ce qui est terminé de ce qui reste à faire.

### Logique
Avant d'écrire la boucle d'entraînement, il fallait fixer les choix de l'expérience dans un fichier YAML.  
Le YAML sert de source de vérité : le futur code d'entraînement devra lire cette config au lieu de coder les hyperparamètres en dur.

### Résultat
La première expérience audio-only est cadrée, même si le loader YAML et la boucle d'entraînement restent à implémenter.

---

## Juin 2026 - Baseline modèle audio

### Réalisations
- Création de `src/models/audio_models.py`.
- Implémentation de `AudioCNNBaseline`.
- Implémentation de blocs convolutionnels simples :
  - convolution 2D,
  - batch normalization,
  - ReLU,
  - max pooling.
- Ajout d'une tête de classification dense.
- Ajout de `build_audio_model()` pour construire le modèle depuis la section `model` du YAML.
- Ajout de docstrings au format Google.
- Création de tests dans `tests/models/test_audio_models.py`.

### Logique
La baseline audio-only doit rester volontairement simple.  
Elle prend en entrée un mel-spectrogramme de forme `[batch_size, channels, n_mels, time_steps]` et produit des logits `[batch_size, num_classes]`.

Ce modèle sert de première brique testable avant d'ajouter :
- la conversion waveform -> mel-spectrogram,
- la boucle d'entraînement,
- les métriques d'évaluation.

### Résultat
Le modèle audio baseline est implémenté et testé.  
Les tests vérifient la forme des sorties, la construction depuis config et le rejet des entrées au mauvais format.

---

## Juin 2026 - Features audio mel-spectrogram

### Réalisations
- Création de `src/data/audio_feature.py`.
- Implémentation de `MelSpectrogramExtractor`.
- Conversion waveform -> spectrogramme via `torch.stft`.
- Construction d'une banque de filtres mel triangulaires sans dépendance externe.
- Support des entrées :
  - `[channels, samples]`
  - `[batch_size, channels, samples]`
- Ajout de `build_audio_feature_extractor()` pour construire l'extracteur depuis la config YAML.
- Ajout de tests dans `tests/data/test_audio_feature.py`.

### Logique
Le dataset charge actuellement l'audio sous forme de waveform brut.  
Le modèle audio CNN, lui, attend un tenseur image-like : `[batch_size, channels, n_mels, time_steps]`.

Cette étape relie donc la couche data au modèle en transformant l'audio brut en représentation temps/fréquence exploitable par le CNN.

Le choix de rester sur `torch.stft` évite d'ajouter immédiatement une dépendance comme `librosa` ou `torchaudio`.  
Cela garde la baseline légère pour une première version.

### Résultat
La chaîne audio-only dispose maintenant de deux briques testées :
- waveform -> mel-spectrogram,
- mel-spectrogram -> logits real/fake.

---

## Juin 2026 - Loader et validation de configuration

### Réalisations
- Création de `src/config.py`.
- Ajout de `load_config()` pour lire les fichiers YAML.
- Ajout de `validate_audio_baseline_config()` pour valider la config audio-only.
- Ajout de `PyYAML` dans `requirements.txt`.
- Ajout de tests dans `tests/test_config.py`.
- Validation du vrai fichier `configs/baseline_audio.yaml` dans les tests.

### Logique
La configuration doit devenir la source de vérité du training.  
Avant d'écrire la boucle d'entraînement, il faut donc s'assurer que le code peut lire le YAML et refuser rapidement une config incomplète ou incohérente.

Le loader reste volontairement simple :
- lecture du fichier YAML,
- vérification des sections obligatoires,
- vérification des clés essentielles,
- validation des types d'expérience supportés pour cette première baseline.

### Résultat
Le projet peut maintenant charger et valider la configuration audio baseline avant de construire l'extracteur audio et le modèle.

---

## Juin 2026 - Trainer audio-only minimal

### Réalisations
- Création de `src/training/trainer.py`.
- Ajout de `resolve_device()` pour gérer `auto`, `cpu` et `cuda`.
- Ajout de `build_optimizer()` pour construire Adam ou SGD depuis la config.
- Ajout de `train_one_epoch()` pour entraîner le modèle audio sur une epoch.
- Ajout de `evaluate_audio_model()` pour mesurer loss et accuracy sans mise à jour des poids.
- Ajout de `EpochMetrics` pour retourner loss, accuracy, nombre de samples et nombre de batches.
- Ajout de tests dans `tests/training/test_trainer.py`.
- Smoke test réel sur deux batches depuis `data/manifests/train_manifest.csv`.

### Logique
L'objectif était de vérifier que les briques audio-only communiquent réellement entre elles :
- DataLoader,
- waveform audio,
- extracteur mel-spectrogram,
- modèle CNN,
- loss,
- optimizer.

La boucle reste volontairement minimale.  
Elle ne gère pas encore les runs, checkpoints, métriques complètes ou fichiers de prédictions.

### Résultat
Un mini entraînement audio-only fonctionne sur des clips réels prétraités.  
La chaîne peut désormais faire un passage complet avec `loss.backward()` et `optimizer.step()`.

---

## Juin 2026 - Organisation des runs expérimentaux

### Réalisations
- Création de `src/runs.py`.
- Ajout de `RunContext` pour centraliser tous les chemins d'un run.
- Ajout de `GitSnapshot` pour capturer branche, commit et état dirty du dépôt.
- Ajout de `generate_run_id()` avec format date, nom d'expérience et commit court.
- Ajout de `create_run_context()` pour créer automatiquement :
  - `runs/<experiment>/<run_id>/`,
  - `config.yaml`,
  - `command.txt`,
  - `git.json`,
  - `metadata.json`,
  - `checkpoints/`,
  - `logs/`,
  - `metrics/`,
  - `predictions/`.
- Mise à jour de `configs/baseline_audio.yaml` pour pointer les sorties vers `runs/`.
- Ajout de tests dans `tests/test_runs.py`.

### Logique
Avant d'ajouter l'évaluation et les checkpoints, il fallait définir où chaque exécution écrit ses résultats.  
Le but est qu'un run soit reproductible et inspectable sans ambiguïté :
- quelle config a été utilisée,
- quelle commande a été lancée,
- sur quel commit,
- où sont les logs, métriques, prédictions et checkpoints.

### Résultat
La structure de runs est prête pour les prochaines étapes.  
Les futurs modules d'évaluation et de checkpointing pourront écrire dans les dossiers fournis par `RunContext`.

---

## Juin 2026 - Entrée CLI `main.py train`

### Réalisations
- Ajout de la sous-commande `train` dans `main.py`.
- Support des options :
  - `--config`,
  - `--epochs`,
  - `--max-batches`,
  - `--batch-size`,
  - `--run-id`,
  - `--runs-root`,
  - `--device`.
- Branchement de la config YAML, du `RunContext`, du DataLoader, de l'extracteur audio, du modèle et du trainer.
- Écriture de `train_metrics.json` dans le dossier `metrics/` du run.
- Ajout d'un mode audio-only dans le DataLoader pour éviter de charger les frames vidéo pendant l'entraînement audio.
- Ajout de tests d'intégration dans `tests/test_main.py`.
- Smoke test réel avec :
  - `main.py train`,
  - CPU,
  - batch size 1,
  - une epoch,
  - un batch.

### Logique
Le premier essai sur CPU a été tué par le système, probablement à cause de la mémoire.  
La cause principale était que le training audio-only chargeait aussi les frames vidéo via le dataset complet.

Le correctif a consisté à garder le comportement multimodal par défaut, mais à permettre au training audio-only de demander uniquement :
- `audio`,
- `label`,
- métadonnées.

Cela rend le smoke train beaucoup plus léger et compatible avec un noeud CPU.

### Résultat
La commande suivante fonctionne :

```bash
python3 main.py train --config configs/baseline_audio.yaml --epochs 1 --max-batches 1 --batch-size 1 --device cpu
```

Elle crée un run et écrit les métriques d'entraînement dans `runs/.../metrics/train_metrics.json`.

---

## Prochaine période prévue - Training audio-only

### Objectifs
- Ajouter validation et checkpoints.
- Ajouter les premières métriques d'évaluation.

### Logique
La prochaine étape consiste à relier les briques existantes :
- manifests,
- dataset,
- features audio,
- modèle,
- config YAML.

Une fois ces éléments connectés, le projet pourra lancer une première expérience audio-only complète.
