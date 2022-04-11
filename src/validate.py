import torch.optim
from tqdm import tqdm

from config import ArgParser, TrainingConfig, build_optimizer
from models import ModelInfoTag, Model, build_model_of, ModelIOHelper
from src.dataloaders import MonoMSWCDataset, DataLoaderMode, ClassificationDataLoader, \
    FewShotDataLoader, DataLoader
from src.models import FewShotModel, CoreModel
from src.trainers.handlers.validators import estimate_accuracy_with_errors
from trainers.handlers import ModelSaver, ClassificationValidator, Printer, PrinterHandler, \
    ValidationMode
from trainers.trainer import Trainer, DefaultTrainer, TrainingParams
from paths import PATH_TO_SAVED_MODELS, PATH_TO_MSWC_WAV

torch.backends.cudnn.benchmark = True

args = ArgParser().parse_args()
config = TrainingConfig({"language": None}).load_json(args.config_path)

language = config["language"]
info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'])
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
model: Model
#
# train_loader: DataLoader = ClassificationDataLoader(
#     MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING), config['batch_size'])

validation_loader: DataLoader = ClassificationDataLoader(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION, is_wav=False),
    config['batch_size'])

# train_loader: DataLoader = \
#     FewShotDataLoader(MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING),
#                       MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING),
#                       config['batch_size'], target='dos', target_probability=0.1)
# validation_loader: DataLoader = \
#     FewShotDataLoader(MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION),
#                       MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION),
#                       config['batch_size'], target='dos', target_probability=0.1)

output_channels = len(validation_loader.get_labels())
print(output_channels)

if config['load_model_from_file']:
    if config['checkpoint_version'] is None:
        raise ValueError('Version of model to be loaded is unknown')
    model = model_io.load_model(CoreModel, info_tag, config['checkpoint_version'],
                                kernel_args={"output_channels": output_channels})
# else:
# model: Model = build_model_of(CoreModel, info_tag,
#                               kernel_args={"output_channels": output_channels})

val_accuracy, val_errors = estimate_accuracy_with_errors(model, validation_loader,
                                                         config['batches_per_validation'])

# train_accuracy, train_errors = estimate_accuracy_with_errors(model, train_loader,
#                                                              config['batches_per_validation'])
#
val_errors = list(sorted(val_errors.items(), key=lambda item: -item[1]))
# train_errors = list(sorted(train_errors.items(), key=lambda item: -item[1]))
#
val_labels = validation_loader.get_labels()
with open('val_errors.txt', 'w') as output:
    output.write("label -> models output: count\n")
    output.write(f"validation accuracy: {val_accuracy}\n\n")
    for idx, count in tqdm(val_errors):
        output.write(f'{val_labels[idx[1]]} -> {val_labels[idx[0]]}: {count}\n')
#
# train_labels = train_loader.get_labels()
# with open('train_errors.txt', 'w') as output:
#     output.write("label -> models output: count\n")
#     output.write(f"train accuracy: {train_accuracy}\n\n")
#     for idx, count in tqdm(train_errors):
#         output.write(f'{train_labels[idx[1]]} -> {train_labels[idx[0]]}: {count}\n')
#
# # if not config['load_optimizer_from_file']:
# #     optimizer: torch.optim.Optimizer = build_optimizer(model, config['optimizer'],
# #                                                        config['optimizer_parameters'])
# #     model.optimizer = optimizer
# #
# # printer: Printer = Printer(config['epochs'], config['batches_per_epoch'])
# # printer_handler: PrinterHandler = PrinterHandler(printer)
# # validation_validator: ClassificationValidator = \
# #     ClassificationValidator(validation_loader,
# #                             batch_count=config['batches_per_validation'],
# #                             mode=ValidationMode.VALIDATION)
# # training_validator: ClassificationValidator = \
# #     ClassificationValidator(train_loader,
# #                             batch_count=config['batches_per_validation'],
# #                             mode=ValidationMode.TRAINING)
# # trainer: Trainer = DefaultTrainer([], [printer_handler],
# #                                   [validation_validator, training_validator,
# #                                    ModelSaver(model_io, 5),
# #                                    printer_handler])
# #
# # training_params: TrainingParams = TrainingParams(batch_count=config['batches_per_epoch'],
# #                                                  batch_size=config['batch_size'],
# #                                                  epoch_count=config['epochs'])
# #
# # trainer.train(model, train_loader, training_params)
