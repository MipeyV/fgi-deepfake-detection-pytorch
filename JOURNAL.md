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

## Juin 2026 - Évaluation audio-only

### Réalisations
- Création de `src/evaluation/metrics.py`.
- Ajout des métriques binaires :
  - accuracy,
  - precision,
  - recall,
  - F1-score,
  - AUC-ROC,
  - confusion matrix.
- Création de `src/evaluation/evaluator.py`.
- Ajout d'une boucle d'inférence audio-only.
- Export des prédictions par clip dans un CSV.
- Export des métriques globales dans un JSON.
- Ajout de la sous-commande `main.py eval`.
- Ajout de tests dans :
  - `tests/evaluation/test_metrics.py`,
  - `tests/evaluation/test_evaluator.py`,
  - `tests/test_main.py`.
- Smoke test réel avec `main.py eval` sur le split train.

### Logique
Le trainer donne une métrique rapide pendant l'apprentissage, mais l'évaluation doit produire des fichiers comparables entre modèles.  
Le format choisi sépare :
- les métriques globales dans `metrics/*.json`,
- les prédictions détaillées dans `predictions/*.csv`.

Chaque ligne de prédiction garde les métadonnées importantes :
- `video_id`,
- `clip_id`,
- `clip_path`,
- label réel,
- label prédit,
- probabilités real/fake,
- indicateur `correct`.

### Résultat
La baseline audio-only peut maintenant être évaluée avec :

```bash
python3 main.py eval --config configs/baseline_audio.yaml --split train --max-batches 1 --batch-size 1 --device cpu
```

La commande écrit :
- `runs/.../metrics/<split>_metrics.json`,
- `runs/.../predictions/<split>_predictions.csv`.

---

## Juin 2026 - Plots de training

### Réalisations
- Création de `src/evaluation/plots.py`.
- Ajout de `load_training_history()` pour lire `train_metrics.json`.
- Ajout de `plot_training_history_svg()` pour générer une courbe SVG.
- Ajout de `plot_metric_history_svg()` pour tracer une ou plusieurs séries par epoch.
- Ajout de `plot_confusion_matrix_svg()` pour visualiser les résultats d'évaluation.
- Ajout du dossier `plots/` dans chaque `RunContext`.
- Branchement automatique après `main.py train` et `main.py eval`.
- Ajout de tests dans `tests/evaluation/test_plots.py`.
- Smoke test réel avec génération de `runs/.../plots/training_history.svg`.
- Smoke test réel avec génération de `runs/.../plots/train_loss.svg`.
- Smoke test réel avec génération de `runs/.../plots/train_accuracy.svg`.
- Smoke test réel avec génération de `runs/.../plots/<split>_confusion_matrix.svg`.

### Logique
Les runs servent maintenant de mini système de tracking d'expériences.  
Après les métriques JSON, il est utile de pouvoir inspecter rapidement l'évolution du modèle au fil des epochs.

Le choix du SVG généré sans dépendance externe garde le projet léger et compatible avec les environnements cluster/headless.

### Résultat
Chaque run d'entraînement peut maintenant produire :
- `metrics/train_metrics.json`,
- `plots/training_history.svg`.
- `plots/train_loss.svg`.
- `plots/train_accuracy.svg`.

Quand une validation est exécutée pendant le training, le run produit aussi :
- `plots/loss_train_vs_val.svg`,
- `plots/accuracy_train_vs_val.svg`.

Chaque run d'évaluation peut aussi produire :
- `metrics/<split>_metrics.json`,
- `predictions/<split>_predictions.csv`,
- `plots/<split>_confusion_matrix.svg`.

Cela permet de visualiser rapidement la loss, l'accuracy et les erreurs real/fake.

---

## Juin 2026 - Checkpoints training audio-only

### Réalisations
- Création de `src/training/checkpoints.py`.
- Ajout de la sauvegarde de checkpoints PyTorch `.pt`.
- Sauvegarde automatique de :
  - `checkpoints/last.pt`,
  - `checkpoints/best.pt`.
- Stockage dans chaque checkpoint :
  - epoch,
  - poids du modèle,
  - état de l'optimizer,
  - métriques de l'epoch,
  - config utilisée,
  - métrique de sélection du meilleur checkpoint.
- Branchement de `main.py train` sur le checkpointing.
- Branchement de `main.py train` sur une validation par epoch quand `val_manifest` existe.
- Mise à jour de `configs/baseline_audio.yaml` pour sélectionner le meilleur checkpoint avec `val_loss`.
- Ajout de tests dans `tests/training/test_checkpoints.py`.
- Mise à jour du test d'intégration `tests/test_main.py` pour vérifier la création de `last.pt` et `best.pt`.

### Logique
L'évaluation produit maintenant des métriques et des prédictions, mais il fallait pouvoir évaluer un modèle réellement entraîné.  
Les checkpoints rendent les runs réutilisables :
- `last.pt` permet de reprendre ou inspecter la dernière epoch,
- `best.pt` permet de conserver automatiquement le meilleur modèle observé.

Le choix de `val_loss` comme métrique de sélection est cohérent avec la boucle actuelle, qui calcule déjà loss et accuracy pendant la validation.

### Résultat
Un entraînement audio-only peut maintenant produire des fichiers de modèle dans :

```bash
runs/.../checkpoints/last.pt
runs/.../checkpoints/best.pt
```

Ces fichiers peuvent ensuite être utilisés par l'évaluation avec :

```bash
python3 main.py eval --config configs/baseline_audio.yaml --split test --checkpoint runs/.../checkpoints/best.pt
```

---

## Juin 2026 - Baseline vidéo-only

### Réalisations
- Création de `configs/baseline_video.yaml`.
- Création de `src/models/video_models.py`.
- Implémentation de `VideoCNNBaseline` avec convolutions 3D.
- Ajout d'une configuration `video.frame_size` pour redimensionner les frames avant entraînement.
- Ajout de `build_frame_resize_transform()` dans `src/data/dataset.py`.
- Ajout de boucles `train_video_one_epoch()` et `evaluate_video_model()`.
- Ajout de `evaluate_video_classifier()` pour produire métriques et prédictions CSV.
- Branchement de `main.py train` et `main.py eval` sur la baseline vidéo via `model.name`.
- Réutilisation des runs, checkpoints, métriques et plots existants.
- Ajout de tests pour le modèle vidéo, le trainer vidéo, l'evaluator vidéo, la config vidéo et l'intégration train/eval vidéo.

### Logique
Avant d'implémenter une approche multimodale ou FGI-inspired, il fallait valider que le pipeline vidéo fonctionne seul.  
Les clips réels contiennent 30 frames en 1920x1080, ce qui rend un training brut trop lourd pour CPU.

La baseline vidéo redimensionne donc les frames à une taille configurable, `64x64` par défaut, afin de tester la chaîne complète sans saturer la mémoire.

### Résultat
La commande suivante fonctionne sur un smoke test CPU :

```bash
python3 main.py train --config configs/baseline_video.yaml --epochs 1 --max-batches 1 --batch-size 1 --device cpu
```

Elle crée un run `baseline-video` avec :
- métriques train/val,
- checkpoints `last.pt` et `best.pt`,
- plots train/val.

Le checkpoint peut ensuite être évalué avec :

```bash
python3 main.py eval --config configs/baseline_video.yaml --split test --checkpoint runs/.../checkpoints/best.pt --max-batches 1 --batch-size 1 --device cpu
```

## Juin 2026 - Setup training cluster et garde-fous

### Réalisations
- Création de `src/training/early_stopping.py`.
- Branchement de l'early stopping dans les boucles audio-only et video-only.
- Suivi de la métrique configurée, actuellement `val_loss`.
- Support de :
  - `training.early_stopping.enabled`,
  - `training.early_stopping.patience`,
  - `training.early_stopping.min_delta`.
- Ajout de l'early stopping dans `configs/baseline_video.yaml`.
- Création de scripts Slurm :
  - `jobs/train_audio_baseline.sbatch`,
  - `jobs/train_video_baseline.sbatch`.
- Scripts paramétrables via variables d'environnement :
  - `CONFIG`,
  - `EPOCHS`,
  - `BATCH_SIZE`,
  - `DEVICE`,
  - `MAX_BATCHES`,
  - `VENV_PATH`.
- Ajout de tests dans `tests/training/test_early_stopping.py`.

### Logique
Avant de lancer des entraînements plus longs sur cluster, il fallait éviter deux risques :
- continuer à entraîner après dégradation de la validation,
- lancer des jobs Slurm non reproductibles ou difficiles à adapter.

L'early stopping complète les checkpoints :
- `best.pt` garde le meilleur modèle selon validation,
- `last.pt` garde le dernier état,
- l'entraînement s'arrête si la métrique suivie ne s'améliore plus.

### Résultat
Les deux baselines peuvent être lancées sur cluster avec :

```bash
sbatch jobs/train_audio_baseline.sbatch
sbatch jobs/train_video_baseline.sbatch
```

Pour un smoke test Slurm court :

```bash
MAX_BATCHES=2 EPOCHS=3 sbatch jobs/train_audio_baseline.sbatch
MAX_BATCHES=2 EPOCHS=3 sbatch jobs/train_video_baseline.sbatch
```

---

## Prochaine période prévue - Baseline multimodale

### Objectifs
- Ajouter une baseline audio-vidéo simple.
- Réutiliser les encodeurs audio et vidéo déjà testés.
- Fusionner les embeddings audio et vidéo par concaténation.
- Comparer audio-only, video-only et audio-video sur les mêmes splits.

### Logique
Le projet dispose maintenant de deux baselines unimodales :
- audio-only,
- video-only.

Avant de passer à une approche FGI-inspired plus fine, il faut créer une baseline multimodale simple.  
Elle servira de référence pour mesurer si la fusion audio-vidéo apporte déjà un gain par rapport aux modèles séparés.

---

## 7 juin 2026 - Premiers runs cluster et fiabilisation du workflow

### Réalisations
- Analyse des runs Slurm audio `842002` et vidéo `842003`.
- Run audio terminé par early stopping à l'epoch 12 :
  - meilleur checkpoint à l'epoch 5,
  - `val_loss = 0.5584`,
  - `val_accuracy = 0.75`.
- Run vidéo interrompu par la limite Slurm de 6 heures à l'epoch 7 :
  - meilleur checkpoint à l'epoch 5,
  - `val_loss = 0.5331`,
  - `val_accuracy = 0.75`.
- Identification d'un biais de classe important :
  - train : 2260 fake / 540 real,
  - validation : 540 fake / 180 real,
  - test : 430 fake / 50 real.
- Ajout de graduations et d'une grille aux plots SVG.
- Sauvegarde de `train_metrics.json` après chaque epoch.
- Passage des scripts Slurm à `python3 -u` pour obtenir les logs sans buffering.
- Ajout de l'évaluation automatique du meilleur checkpoint sur le test en fin
  d'entraînement.
- Ajout de `jobs/eval_baseline.sbatch` pour évaluer un checkpoint indépendamment
  du job d'entraînement.
- Ajout de la reprise d'entraînement avec `main.py train --resume` :
  - restauration des poids,
  - restauration de l'optimizer,
  - reprise à l'epoch suivante,
  - conservation de l'historique et du meilleur checkpoint.
- Ajout de `jobs/eval_ensemble.sbatch` et de `main.py ensemble-eval`.
- Ajout d'une fusion tardive audio-vidéo par moyenne de `prob_fake`.
- Export du taux d'accord, des désaccords par clip et des métriques séparées
  audio, vidéo et ensemble.
- Exclusion de `jobs/logs/` du suivi Git.

### Logique
Le job vidéo a montré qu'un workflow cluster ne doit pas supposer que la boucle
d'entraînement atteindra toujours sa fin. Les métriques, checkpoints et logs
doivent rester exploitables après une interruption Slurm.

Les accuracies observées correspondent presque exactement à la proportion de
la classe `fake`. Elles ne suffisent donc pas à démontrer que les modèles
discriminent réellement les deux classes. La comparaison audio-vidéo doit
mesurer les désaccords et examiner les métriques par classe, pas seulement
l'accuracy.

### Résultat
- Les entraînements peuvent être repris depuis `last.pt` dans un nouveau job.
- Les checkpoints peuvent être évalués dans un job séparé même si le training
  a été interrompu.
- Un job d'évaluation ensemble, `843121`, a été soumis sur la partition 3090.
  Au moment de cette mise à jour, il est en attente de ressources.
- La suite contient 97 tests passants.

### Limites observées
- La patience d'early stopping des anciens checkpoints n'est pas restaurée et
  repart à zéro lors d'une reprise.
- Le déséquilibre des classes doit encore être traité avant de comparer
  sérieusement les performances.
- L'ensemble actuel est une fusion tardive de probabilités, pas encore un modèle
  multimodal entraîné conjointement.
