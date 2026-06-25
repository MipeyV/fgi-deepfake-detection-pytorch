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
  - `experiments/` (remplacé ensuite par `runs/` pour les artefacts locaux)
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

---

## 9 juin 2026 - Traitement du déséquilibre des classes

### Réalisations
- Analyse de l'évaluation croisée audio, vidéo et ensemble sur le test :
  - audio : `AUC = 0.5846`,
  - vidéo : `AUC = 0.6225`,
  - ensemble : `AUC = 0.6331`.
- Identification d'une prédiction systématique de la classe `fake` :
  - 430 vrais positifs,
  - 50 faux positifs,
  - aucun échantillon prédit `real`.
- Ajout de class weights à la cross-entropy pour les trainings audio et vidéo.
- Calcul automatique des poids à partir du manifest train uniquement :
  - `real = 2.593`,
  - `fake = 0.619`.
- Support de poids manuels dans la configuration.
- Validation des poids configurés et détection des classes absentes.
- Activation de `class_weights: balanced` dans les deux configs baseline.
- Ajout de tests unitaires pour le calcul et la construction de la loss.
- Ajout de `notebooks/dataset_static_analysis.ipynb` pour analyser :
  - l'équilibre des classes par split,
  - les distributions au niveau clip et vidéo,
  - le nombre de clips par vidéo,
  - les doublons et les fuites entre splits,
  - les chemins de clips manquants,
  - les poids de classes utilisés pour le training.
- Ajout d'`ipykernel` aux dépendances pour exécuter le notebook dans VS Code.
- Relance d'un training vidéo pondéré sur le cluster avec le job Slurm `845027`.

### Logique
L'accuracy de `89.58 %` correspondait exactement à la proportion de `fake`
dans le test. Elle était donc trompeuse : un modèle prédisant toujours `fake`
obtenait cette accuracy sans reconnaître aucun exemple `real`.

L'AUC supérieure à `0.5` montre qu'un faible signal de classement existe, mais
la frontière de décision est fortement biaisée par la distribution des
classes. La cross-entropy pondérée donne davantage d'importance aux erreurs sur
la classe minoritaire `real`, sans utiliser les données de validation ou de
test pour calculer les poids.

L'audit statique confirme le déséquilibre :
- train : 540 real / 2260 fake,
- validation : 180 real / 540 fake,
- test : 50 real / 430 fake.

### Résultat
- Les trainings audio et vidéo utilisent désormais automatiquement des poids
  équilibrés.
- L'audit ne détecte aucun doublon, aucune fuite de vidéo entre splits, aucun
  label invalide et aucun chemin de clip manquant.
- La suite contient 101 tests passants après l'ajout des class weights.
- Le prochain point de comparaison sera l'AUC et la matrice de confusion des
  nouveaux checkpoints, en particulier le nombre de vrais négatifs retrouvés.

---

## 10 juin 2026 - Passage à R3D-18 et modularisation du pipeline vidéo

### Analyse du training pondéré
- Analyse du job Slurm vidéo `845027`.
- Confirmation de l'utilisation des poids de classes :
  - `real = 2.593`,
  - `fake = 0.619`.
- Résultats sur le test :
  - `accuracy = 0.8104`,
  - `AUC = 0.6164`,
  - `F1 = 0.8926`,
  - 11 vrais négatifs,
  - 39 faux positifs,
  - 52 faux négatifs,
  - 378 vrais positifs.
- Le modèle reconnaît désormais 11 exemples `real` sur 50, contre aucun avant
  la pondération.
- L'AUC reste cependant proche de la baseline précédente (`0.6225`) et baisse
  légèrement malgré la correction partielle du seuil de décision.

### Réalisations
- Ajout d'un classifieur vidéo R3D-18 basé sur Torchvision.
- Support optionnel des poids préentraînés `KINETICS400_V1`.
- Remplacement de la tête Kinetics-400 par une tête de classification binaire.
- Ajout de la normalisation attendue par les poids Kinetics-400.
- Ajout du preprocessing spatial recommandé :
  - redimensionnement en `128 x 171`,
  - crop central en `112 x 112`.
- Ajout de `configs/r3d18_video.yaml`.
- Support du training, de l'évaluation et de l'ensemble avec le modèle R3D-18.
- Conservation du modèle CNN vidéo historique pour permettre les comparaisons.

### Modularisation
- Séparation des architectures vidéo dans `src/models/video/` :
  - `baseline.py`,
  - `r3d18.py`,
  - `factory.py`,
  - validation de configuration dédiée.
- Conservation de `src/models/video_models.py` comme façade compatible avec les
  anciens imports.
- Création de `src/data/video/` pour isoler les pipelines d'entrée :
  - resize carré pour la baseline,
  - resize puis crop central pour R3D-18,
  - factory et validation indépendantes des modèles.
- Modification des configs afin de sélectionner séparément :
  - `video.preprocessing.name`,
  - `model.name`.
- Branchement du training, de l'évaluation et de la fusion tardive sur le même
  pipeline d'entrée configurable.

### Contrats retenus
Les composants vidéo respectent actuellement les interfaces suivantes :
- pipeline d'entrée :
  `frames = [batch, frames, channels, height, width]`,
- modèle :
  `logits = [batch, classes]`.

Le trainer et l'évaluateur ne connaissent donc plus l'architecture du modèle
ni les détails du resize et du crop. Un modèle peut réutiliser plusieurs
pipelines d'entrée, et un pipeline peut être comparé avec plusieurs modèles.

### Logique
Les class weights ont montré que le déséquilibre n'était pas l'unique limite :
ils permettent de récupérer quelques exemples `real`, mais n'améliorent pas
l'AUC. La baseline 3D CNN semble donc manquer de capacité ou de représentations
temporelles suffisamment robustes.

R3D-18 sert de nouvelle baseline vidéo préentraînée avant la construction d'un
modèle multimodal ou FGI-inspired. La modularisation évite cependant de lier le
projet à R3D-18 : une future approche FGI pourra introduire son propre
échantillonnage temporel, des régions faciales, des landmarks ou des entrées
audio-visuelles synchronisées sans modifier les modèles existants.

### Résultat
- La suite contient 109 tests passants.
- Les baselines vidéo et R3D-18 utilisent le même workflow de training et
  d'évaluation.
- Le lancement cluster prévu est :

```bash
sbatch --export=ALL,CONFIG=configs/r3d18_video.yaml \
  jobs/train_video_baseline.sbatch
```

### Limites observées
- Les poids R3D-18 représentent environ 127 Mo et ne sont pas encore présents
  dans le cache Torch local.
- Le premier lancement doit donc disposer d'un accès réseau ou d'un cache
  préalimenté sur le cluster.
- Le pipeline reste unimodal : la synchronisation et les interactions
  audio-vidéo propres à FGI ne sont pas encore implémentées.

---

## 12 juin 2026 - Résultats R3D-18 et préparation du pipeline FGI

### Résultats R3D-18
- Training arrêté à l'epoch 6 après dégradation de la validation.
- Meilleure validation observée :
  - `val_accuracy = 0.7653` à l'epoch 3,
  - meilleur `val_loss = 0.4930` à l'epoch 1.
- Résultats sur les 480 clips de test :
  - `accuracy = 0.7417`,
  - `AUC = 0.6207`,
  - `F1 = 0.8450`,
  - 18 vrais négatifs,
  - 32 faux positifs,
  - 92 faux négatifs,
  - 338 vrais positifs.

### Analyse
R3D-18 améliore légèrement l'AUC et reconnaît davantage de clips `real` que
la baseline vidéo pondérée, mais il perd en accuracy et en F1. La balanced
accuracy reste faible et proche du hasard à cause d'un rappel limité sur la
classe `real`.

L'écart croissant entre l'entraînement et la validation confirme un
surapprentissage. Continuer uniquement à ajuster la tête R3D-18 paraît moins
informatif que de passer à l'objectif multimodal du projet.

### Préprocessing FGI
- Création de la branche `feature/fgi-preprocessing`.
- Ajout du transform configurable `resize_normalize`.
- Support des paramètres YAML :
  - `frame_size`,
  - moyenne RGB `mean`,
  - écart-type RGB `std`.
- Ajout du réglage FGI officiel :
  - resize `224 x 224`,
  - `mean = [0.5, 0.5, 0.5]`,
  - `std = [0.5, 0.5, 0.5]`,
  - valeurs de sortie dans `[-1, 1]`.
- Ajout de `configs/fgi_preprocessing.yaml`.

### Choix d'architecture
Le preprocessing et le modèle restent sélectionnés indépendamment :

```yaml
video:
  preprocessing:
    name: resize_normalize

model:
  name: video_cnn_baseline
```

La config préparatoire utilise volontairement le CNN vidéo existant. Associer
directement cette normalisation à R3D-18 préentraîné appliquerait des
statistiques différentes de celles de Kinetics-400. Le futur modèle
`model.name: fgi` pourra réutiliser le preprocessing sans modifier les
baselines.

### Prochaine étape
- Fournir conjointement les frames et l'audio synchronisés à un trainer
  multimodal.
- Ajouter les encodeurs audio et vidéo.
- Implémenter les distances locales, l'attention spatiale puis les pseudo-fakes
  temporels.

### Résultat
- Les trois preprocessings vidéo sont sélectionnables par configuration :
  `resize_square`, `resize_center_crop`, `resize_normalize`.
- Le choix du preprocessing reste indépendant de `model.name`.
- La suite complète contient 111 tests passants avant l'ajout de la config
  préparatoire.

---

## 12 juin 2026 - Crops faciaux stabilisés pour FGI

### Objectif
Construire le cache facial hors ligne nécessaire avant le modèle multimodal.
Les clips source restent dans `data/processed/`; les nouvelles données sont
écrites dans `data/processed_fgi/` avec des manifests séparés.

### Réalisations
- Ajout de `src/data/fgi_face_crops.py`.
- Ajout de la commande CLI `main.py fgi-face-crops`.
- Détection recommandée avec YuNet et un modèle ONNX explicite.
- Conservation d'un backend Haar pour les tests de développement.
- Pour chaque clip :
  - détection sur chaque frame,
  - association des détections en pistes par IoU,
  - sélection de la piste faciale la plus longue,
  - agrégation médiane des centres et dimensions,
  - ajout d'une marge configurable,
  - application d'un crop carré identique à toutes les frames,
  - resize en `256 x 256` par défaut,
  - copie de `audio.wav`.
- Conservation de toutes les colonnes du manifest source, dont
  `video_id_hashmod10` et `split`.
- Ajout d'un seuil `min_detection_fraction` pour rejeter les clips dont trop
  peu de frames contiennent un visage.
- Ajout des politiques `error` et `skip`.
- Ajout d'une planche contact PNG montrant le début, le milieu et la fin de
  chaque clip échantillonné.

### Contrôle qualité
Un premier essai avec Haar a techniquement produit un crop, mais l'inspection
visuelle a révélé un faux positif sur une personne tournée de dos. Ce résultat
a motivé deux garde-fous :
- YuNet devient le détecteur recommandé et par défaut dans la CLI ;
- le modèle ONNX doit être fourni explicitement.

Le fallback Haar n'est donc pas présenté comme un pipeline de production.
L'absence de visage fiable doit conduire à un rejet explicite ou à un clip
comptabilisé comme ignoré.

Le même clip a ensuite été traité avec le modèle YuNet officiel. La planche
contact début/milieu/fin montre correctement l'occultation initiale puis le
visage suivi sur les frames suivantes. Ce smoke test valide le chemin CLI,
l'inférence ONNX, le tracking, les crops, la copie audio et le manifest.

### Commande

```bash
python3 main.py fgi-face-crops \
  --manifest data/manifests/train_manifest.csv \
  --output-dir data/processed_fgi \
  --output-manifest data/manifests_fgi/train_manifest.csv \
  --detector-model models/face_detection_yunet_2023mar.onnx \
  --missing-face-policy skip \
  --contact-sheet runs/fgi-face-crops/train_contact_sheet.png
```

La même commande devra être exécutée pour validation et test. Les taux de clips
ignorés et les trois planches contact devront être vérifiés avant de modifier
les configs d'entraînement.

### Prochaine étape
- Télécharger le modèle YuNet officiel.
- Générer les trois caches et manifests.
- Auditer visuellement les crops et le taux d'échec.
- Brancher les manifests FGI dans un dataset multimodal.

### Résultat
- Suite complète : 118 tests passants.
- Aucun fichier du dataset source n'est modifié.

---

## 12 juin 2026 - Dataset multimodal synchronisé FGI

### Objectif
Fournir un contrat d'entrée strict aux futurs encodeurs audio et vidéo, sans
implémenter prématurément le modèle.

### Réalisations
- Ajout de `FGIMultimodalDataset`.
- Ajout d'un collate et d'une factory de dataloader dédiés.
- Ajout d'un pipeline construit depuis la configuration.
- Ajout de `configs/fgi_inspired.yaml`.
- Ajout de la commande `main.py fgi-data-smoke`.

### Contrat

```text
frames: [batch, 30, 3, 224, 224]
audio:  [batch, 48000]
label:  [batch]
```

Chaque clip doit contenir exactement :
- 30 images JPEG de visage ;
- un WAV mono PCM 16 bits ;
- 48 000 échantillons à 48 kHz.

Les frames sont normalisées avec `mean=std=0.5`. L'audio utilise la
normalisation min-max par clip du dépôt FGI original. Un signal constant est
converti en zéros afin d'éviter une division par zéro.

### Séparation des responsabilités
Le dataset générique historique reste permissif pour les baselines. Le dataset
FGI est séparé et strict afin qu'un clip incomplet ou désynchronisé échoue
avant d'atteindre le modèle.

La config déclare :

```yaml
model:
  name: fgi_inspired
  implementation_status: pending
```

Le validateur interdit de marquer le modèle prêt tant que son implémentation
n'existe pas.

### Smoke test réel
Le crop YuNet généré précédemment a été chargé avec succès :

```text
frames (1, 30, 3, 224, 224), range [-1, 1]
audio  (1, 48000), range [-1, 1]
label  (1,)
```

### Prochaine étape
- Générer les manifests FGI complets.
- Implémenter les encodeurs audio et vidéo.
- Définir les cartes de features locales comparables avant l'attention.

### Résultat
- Suite complète : 124 tests passants.
- Smoke CLI réel validé avec `main.py fgi-data-smoke`.

---

## 12 juin 2026 - Encodeurs audio et vidéo FGI

### Référence
Le modèle FGI original produit :
- des features vidéo locales `[B, 128, 15, 28, 28]` ;
- des features audio temporelles `[B, 128, 15]`.

Ce contrat est conservé, mais les composants sont réimplémentés de manière
modulaire et testable.

### Encodeur vidéo
- Réseau résiduel 3D compact.
- Entrée `[B, 30, 3, 224, 224]`.
- Réorganisation interne en `[B, 3, 30, 224, 224]`.
- Réduction temporelle et spatiale progressive.
- Pooling adaptatif final vers `[B, 128, 15, 28, 28]`.

### Encodeur audio
- Entrée waveform brute `[B, 48000]`.
- Convolution initiale `kernel=80`, `stride=8`, comme dans FGI.
- Pooling et convolutions temporelles.
- Projection vers 128 canaux.
- Pooling adaptatif final vers `[B, 128, 15]`.

### Alignement
`FGIEncoderPair` vérifie que les deux encodeurs partagent :
- la même dimension d'embedding ;
- le même nombre de positions temporelles.

La configuration contient maintenant :

```yaml
model:
  implementation_status: encoders_ready
  encoders:
    embedding_dim: 128
    temporal_size: 15
    spatial_size: 28
```

### Tests
- Tests de shapes sur des tenseurs synthétiques.
- Vérification de propagation des gradients jusqu'aux deux entrées.
- Validation de la factory depuis la vraie configuration.
- Rejet des dimensions audio/vidéo incompatibles.
- Commande `main.py fgi-encoder-smoke`.

### Smoke test réel
Le clip YuNet de contrôle traverse le dataloader puis les encodeurs sur CPU :

```text
video features shape=(1, 128, 15, 28, 28)
audio features shape=(1, 128, 15)
```

### Limite
Ces features ne produisent pas encore une prédiction. La prochaine étape
calculera les distances audio-visuelles locales, puis l'attention spatiale et
la tête de classification.

### Résultat
- Suite complète : 130 tests passants.
- Smoke encodeurs validé sur un crop YuNet réel.

---

## 12 juin 2026 - Modèle de classification FGI

### Calcul d'incohérence locale
Les features audio `[B, D, T]` sont comparées à chaque position spatiale des
features vidéo `[B, D, T, H, W]`. La distance euclidienne sur les axes
d'embedding et temporel produit une carte `[B, H, W]`.

### Attention et classification
- Projections audio et vidéo configurables pour calculer une attention
  spatiale normalisée.
- Modes d'attention `multiply` et `residual`.
- Pondération de la carte d'incohérence.
- `LayerNorm`, dropout configurable et couche linéaire vers deux logits.
- Sortie structurée exposant logits, cartes et features intermédiaires.

### Configuration et validation
La configuration est marquée `model_ready` et contient les dimensions
d'attention, son mode, le dropout et la stabilité numérique de la distance.
La commande `fgi-model-smoke` exécute le chemin complet du manifest aux
logits.

### Tests
- Valeur exacte de la distance locale sur un exemple contrôlé.
- Shapes des logits, cartes d'incohérence, attention et features.
- Normalisation de l'attention.
- Fonctionnement sans attention.
- Propagation des gradients dans les deux modalités.
- Construction depuis la configuration du projet.

### Limite
Le forward complet existe désormais. Les commandes génériques `train` et
`eval` ne prennent pas encore en charge `fgi_inspired`.

### Résultat
- Suite complète : 136 tests passants.
- Smoke réel validé du manifest aux logits sur CPU.
- Sorties observées : logits `(1, 2)`, incohérence `(1, 28, 28)` et attention
  `(1, 28, 28)`.

---

## 12 juin 2026 - Entraînement et évaluation FGI

### Entraînement
- Ajout d'une boucle multimodale consommant simultanément les frames de visage
  et l'audio brut.
- Support de la cross-entropy pondérée, de la validation, de l'early stopping
  et de la limite `--max-batches`.
- Sauvegarde des checkpoints `best.pt` et `last.pt`.
- Reprise d'entraînement avec `--resume`.

### Évaluation
- Support de `fgi_inspired` dans la commande générique `eval`.
- Export des probabilités réel/fake et des prédictions par clip.
- Calcul des métriques binaires et génération de la matrice de confusion.
- Évaluation automatique du meilleur checkpoint après entraînement.

### Cluster
Le job `jobs/train_fgi.sbatch` expose les variables `EPOCHS`, `BATCH_SIZE`,
`MAX_BATCHES`, `RUN_ID`, `RESUME` et `DEVICE`.

### Validation
Un test d'intégration entraîne un petit modèle FGI, écrit les artefacts du run,
recharge le meilleur checkpoint et exécute une évaluation séparée.

Un smoke CPU sur le clip YuNet réel a également validé un backward complet,
la mise à jour de l'optimizer, l'écriture du checkpoint et les courbes. Ce
smoke a permis de corriger la sélection de la métrique d'early stopping
lorsqu'aucun manifest de validation n'est disponible.

### Résultat
- Suite complète : 141 tests passants.
- Entraînement et évaluation FGI disponibles depuis `main.py`.
- Job Slurm FGI prêt pour un smoke GPU puis l'expérience complète.

---

## 13 juin 2026 - Premier résultat complet FGI

### Run
Le run `20260612-164807_fgi-inspired_5932f5e` a été entraîné avec la
configuration FGI complète. L'early stopping s'est déclenché à l'époque 41.
Le meilleur checkpoint selon la loss de validation est celui de l'époque 33 :

- accuracy train : `0.9449` ;
- accuracy validation : `0.8593` ;
- loss validation : `0.3160`.

### Test
L'évaluation du meilleur checkpoint sur 436 clips donne :

- AUC ROC : `0.8254` ;
- accuracy : `0.8372` ;
- F1 : `0.9067` ;
- precision : `0.9200` ;
- recall : `0.8938`.

La matrice de confusion contient 20 vrais négatifs, 30 faux positifs,
41 faux négatifs et 345 vrais positifs.

### Comparaison
L'AUC progresse nettement par rapport aux expériences vidéo précédentes :

- baseline vidéo : `0.6164` ;
- R3D-18 : `0.6207` ;
- FGI : `0.8254`.

### Interprétation
Le split de test est déséquilibré avec 386 clips fake et 50 clips real.
L'accuracy brute doit donc être interprétée avec prudence : au seuil `0.5`,
le rappel fake atteint `89.4 %`, mais seulement `40.0 %` des clips real sont
correctement reconnus. L'accuracy équilibrée correspondante est `0.6469`.

L'AUC est le résultat principal de ce run : le modèle sépare bien mieux les
classes que les baselines, mais le seuil de décision doit être calibré sur la
validation. Une évaluation agrégée par vidéo et des métriques équilibrées
devront compléter la prochaine expérience.

---

## 13 juin 2026 - Calibration et évaluation par vidéo

### Calibration
Le seuil de décision peut maintenant être appris exclusivement sur le split
de validation. La configuration FGI sélectionne le seuil qui maximise
l'accuracy équilibrée, puis réutilise ce seuil sans modification sur le test.
Une valeur explicite passée avec `--decision-threshold` reste prioritaire.

Le fichier `metrics/threshold_calibration.json` conserve le seuil choisi, la
métrique optimisée, sa valeur sur validation et le nombre d'exemples utilisés.
Une calibration complète exige la présence des deux classes. Les smoke tests
limités par `--max-batches` signalent une validation incomplète et utilisent
temporairement le seuil `0.5`.

### Métriques
Les rapports binaires incluent désormais :

- accuracy et accuracy équilibrée ;
- precision, recall et spécificité ;
- F1 fake et F1 macro ;
- AUC ROC et average precision ;
- seuil de décision appliqué.

### Niveau vidéo
Les probabilités fake des clips partageant le même `video_id` sont moyennées.
Le pipeline écrit séparément :

- `predictions/<split>_video_predictions.csv` ;
- `metrics/<split>_video_metrics.json`.

Cette agrégation évite qu'une vidéo découpée en beaucoup de clips pèse
artificiellement plus lourd dans le résultat final.

### Validation
- Suite complète : 146 tests passants.
- Vérification syntaxique de `main.py` et des modules d'évaluation.

---

## 25 juin 2026 - CI, documentation dataset et analyse des durées

### Qualité et CI
- Ajout de la configuration Ruff dans `pyproject.toml` :
  - lint `E`, `F`, `I`, `UP`, `B`,
  - formatage Ruff,
  - exclusion des notebooks du lint/format automatique.
- Ajout du workflow GitHub Actions `.github/workflows/ci.yml`.
- Le job `Lint, format, and test` installe les dépendances système nécessaires
  (`ffmpeg`, `libgl1`), installe les dépendances Python, puis exécute :
  - `ruff check .`,
  - `ruff format --check .`,
  - `python -m compileall -q main.py src tests`,
  - `pytest`.
- La protection de branche attend désormais ce check avant intégration par PR.

### Organisation du code
- Réorganisation des modèles en sous-packages plus explicites :
  - `src/models/baselines/audio/`,
  - `src/models/baselines/video/`,
  - `src/models/fgi/`.
- Séparation plus nette des factories, validations et implémentations vidéo.
- Adaptation des imports dans le code, les tests et les configurations.

### Documentation README
- Ajout d'une section de préparation complète du dataset DFDC :
  - aller sur Kaggle,
  - créer un compte ou se connecter,
  - vérifier son identité Kaggle si nécessaire,
  - rejoindre la compétition et accepter les règles,
  - télécharger et extraire les données,
  - préparer les dossiers `real/` et `fake`,
  - lancer le preprocessing audio-vidéo,
  - générer les manifests train/validation/test.
- Ajout d'une étape explicite d'installation de FFmpeg, requis par le
  preprocessing pour normaliser les vidéos, extraire les frames et extraire
  l'audio.
- Ajout d'une note sur les paramètres par défaut du preprocessing :
  `30` frames à `30` FPS donnent des clips de `1` seconde.

### Analyse dataset
Le journal mentionnait déjà le notebook `dataset_static_analysis.ipynb` et le
nombre de clips par vidéo, mais pas encore l'analyse explicite des durées.

- Ajout d'une section `Estimated video durations` dans
  `notebooks/dataset_static_analysis.ipynb`.
- Les durées sont estimées à partir du nombre de clips prétraités, sans relire
  les vidéos brutes.
- Avec les paramètres actuels, chaque clip vaut `1s`.
- Vérification sur les manifests courants :
  - toutes les vidéos ont exactement `10` clips,
  - les vidéos du subset analysé durent donc environ `10s`,
  - la durée estimée est uniforme entre `real` et `fake`.
- Le résumé compact du notebook affiche maintenant :
  - la durée moyenne estimée par classe,
  - le range global de durée,
  - le flag d'uniformité.

### Suivi projet
- Mise à jour de `SUIVI.md` avec les changements depuis le 13 juin :
  CI/Ruff, refactor modèles, documentation dataset, analyse des durées et
  prochaines étapes.
- Passage du statut global à `5/5 (100%)` pour les modules principaux déjà
  implémentés, tout en conservant les améliorations expérimentales restantes
  dans les prochaines étapes.
