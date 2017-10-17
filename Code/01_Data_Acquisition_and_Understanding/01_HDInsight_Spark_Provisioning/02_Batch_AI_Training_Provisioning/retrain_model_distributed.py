'''
retrain_model_distributed.py
by Mary Wahl, 2017
Copyright Microsoft, all rights reserved

Retrain AlexNet and ResNet 18 models to classify aerial images by land use.
Makes use of distributed learners.
Expects the following parameters:
- input_dir:         The parent directory containing training images. This
                     directory should contain only subdirectories (whose names
                     will be used as the class labels). Each subdirectory should
                     contain only image files and should not be empty.
- validation_dir:    The parent directory containing validation images, similar
				     in contents to input_dir.
- output_model_name: The filepath where the retrained model will be stored.
                     Supporting files will be stored to the same directory.
- model_path:        The location of the pretrained AlexNet or ResNet 18 model
- retraining_type:   Must be "last_only", "fully_connected", or "all". Cannot
                     use retraining type "fully_connected" with model type
                     "resnet18"
- model_type:        Must be "alexnet" or "resnet18"

Side effects:
This script will create a temporary directory, in which it will write MAP
files The directory will be removed on completion.
'''

import numpy as np
import pandas as pd
import os, argparse, glob, tempfile, cntk
from cntk.io import transforms as xforms
import cntk.train.distributed as distributed
from cntk.train.training_session import CheckpointConfig, training_session
from PIL import Image


def write_map_file(map_filename, input_dir, output_dir):
	'''
	Writes the map file required by ImageDeserializer. Returns the number of
	distinct classes found in the training set.
	'''
	df = pd.DataFrame([])
	df['filename'] = list(glob.iglob(os.path.join(input_dir, '*', '*')))
	df['label'] = df['filename'].apply(lambda x:
									   os.path.basename(os.path.dirname(x)))
	labels = list(np.sort(df['label'].unique().tolist()))
	with open(os.path.join(output_dir, 'labels_to_inds.tsv'), 'w') as f:
		for i, label in enumerate(labels):
			f.write('{}\t{}\n'.format(label, i))
	df['idx'] = df['label'].apply(lambda x: labels.index(x))
	df = df[['filename', 'idx']].sample(frac=1)
	df.to_csv(map_filename, index=False, sep='\t', header=False)
	return(len(labels), len(df.index))


def create_minibatch_source(map_filename, num_classes):
	transforms = [xforms.crop(crop_type='randomside',
										  side_ratio=0.85,
										  jitter_type='uniratio'),
				  xforms.scale(width=224,
	  						   height=224,
	  						   channels=3,
	  						   interpolations='linear'),
				  xforms.color(brightness_radius=0.2,
	  						   contrast_radius=0.2,
	  						   saturation_radius=0.2)]
	return(cntk.io.MinibatchSource(cntk.io.ImageDeserializer(
		map_filename,
		cntk.io.StreamDefs(
			features=cntk.io.StreamDef(
				field='image', transforms=transforms, is_sparse=False),
			labels=cntk.io.StreamDef(
				field='label', shape=num_classes, is_sparse=False)))))


def load_alexnet_model(image_input, num_classes, model_filename,
					   retraining_type):
	''' Load pretrained AlexNet for desired level of retraining '''
	loaded_model = cntk.load_model(model_filename)

	# Load the convolutional layers, freezing if desired
	feature_node = cntk.logging.graph.find_by_name(loaded_model, 'features')
	last_conv_node = cntk.logging.graph.find_by_name(loaded_model, 'conv5.y')
	conv_layers = cntk.ops.combine([last_conv_node.owner]).clone(
		cntk.ops.functions.CloneMethod.clone if retraining_type == 'all' \
			else cntk.ops.functions.CloneMethod.freeze,
		{feature_node: cntk.ops.placeholder()})

	# Load the fully connected layers, freezing if desired
	last_node = cntk.logging.graph.find_by_name(loaded_model, 'h2_d')
	fully_connected_layers = cntk.ops.combine([last_node.owner]).clone(
		cntk.ops.functions.CloneMethod.freeze if retraining_type == \
			'last_only' else cntk.ops.functions.CloneMethod.clone,
		{last_conv_node: cntk.ops.placeholder()})

	# Define the network using the loaded layers
	feat_norm = image_input - cntk.layers.Constant(114)
	conv_out = conv_layers(feat_norm)
	fc_out = fully_connected_layers(conv_out)
	new_model = cntk.layers.Dense(shape=num_classes, name='last_layer')(fc_out)
	return(new_model)


def load_resnet18_model(image_input, num_classes, model_filename,
					   retraining_type):
	''' Load pretrained ResNet18 for desired level of retraining '''

	# Load existing layers, freezing as desired
	loaded_model = cntk.load_model(model_filename)
	feature_node = cntk.logging.graph.find_by_name(loaded_model, 'features')
	last_node = cntk.logging.graph.find_by_name(loaded_model, 'z.x')
	cloned_layers = cntk.ops.combine([last_node.owner]).clone(
		cntk.ops.functions.CloneMethod.freeze if retraining_type == \
			'last_only' else cntk.ops.functions.CloneMethod.clone,
		{feature_node: cntk.ops.placeholder()})

	# Define the network using the loaded layers
	feat_norm = image_input - cntk.layers.Constant(114)
	cloned_out = cloned_layers(feat_norm)
	new_model = cntk.layers.Dense(num_classes)(cloned_out)
	return(new_model)


def retrain_model(map_filename, output_dir, num_classes, epoch_size,
				  model_filename, num_epochs, model_type, retraining_type):
	''' Coordinates retraining after MAP file creation '''

	# load minibatch and model
	minibatch_source = create_minibatch_source(map_filename, num_classes)

	image_input = cntk.ops.input_variable((3, 224, 224))
	label_input = cntk.ops.input_variable((num_classes))
	input_map = {image_input: minibatch_source.streams.features,
				 label_input: minibatch_source.streams.labels}

	if model_type == 'alexnet':
		model = load_alexnet_model(image_input, num_classes, model_filename,
								   retraining_type)
	elif model_type == 'resnet18':
		model = load_resnet18_model(image_input, num_classes, model_filename,
								    retraining_type)

	# Set learning parameters
	ce = cntk.losses.cross_entropy_with_softmax(model, label_input)
	pe = cntk.metrics.classification_error(model, label_input)
	l2_reg_weight = 0.0005
	lr_per_sample = [0.00001] * 33 + [0.000001] * 33 + [0.0000001]
	momentum_time_constant = 10
	mb_size = 16
	lr_schedule = cntk.learners.learning_rate_schedule(lr_per_sample,
		unit=cntk.UnitType.sample)
	mm_schedule = cntk.learners.momentum_as_time_constant_schedule(
		momentum_time_constant)

	# Instantiate the appropriate trainer object
	my_rank = distributed.Communicator.rank()
	num_workers = distributed.Communicator.num_workers()
	num_minibatches = int(np.ceil(epoch_size / mb_size))

	progress_writers = [cntk.logging.progress_print.ProgressPrinter(
		tag='Training',
		num_epochs=num_epochs,
		freq=num_minibatches,
		rank=my_rank)]
	learner = cntk.learners.fsadagrad(parameters=model.parameters,
									  lr=lr_schedule,
									  momentum=mm_schedule,
									  l2_regularization_weight=l2_reg_weight)
	if num_workers > 1:
		parameter_learner = distributed.data_parallel_distributed_learner(
			learner, num_quantization_bits=32)
		trainer = cntk.Trainer(model, (ce, pe), parameter_learner,
							   progress_writers)
	else:
		trainer = cntk.Trainer(model, (ce, pe), learner, progress_writers)

	# Print summary lines to stdout and perform training
	if my_rank == 0:
		print('Retraining model for {} epochs.'.format(num_epochs))
		print('Found {} workers'.format(num_workers))
		print('Printing progress every {} minibatches'.format(num_minibatches))
		cntk.logging.progress_print.log_number_of_parameters(model)

	training_session(
		trainer=trainer,
		max_samples=num_epochs * epoch_size,
		mb_source=minibatch_source, 
		mb_size=mb_size,
		model_inputs_to_streams=input_map,
		checkpoint_config=CheckpointConfig(
			frequency=epoch_size,
			filename=os.path.join(output_dir, 'retrained_checkpoint.model')),
		progress_frequency=epoch_size
	).train()

	distributed.Communicator.finalize()
	if my_rank == 0:
		trainer.model.save(os.path.join(output_dir, 'retrained.model'))

	return(my_rank)


def evaluate_model(map_filename, output_dir, num_classes):
	''' Evaluate the model on the test set, storing predictions to a file '''
	inds_to_labels = {}
	with open(os.path.join(output_dir, 'labels_to_inds.tsv'), 'r') as f:
		for line in f:
			label, ind = line.strip().split('\t')
			inds_to_labels[int(ind)] = label

	loaded_model = cntk.load_model(os.path.join(output_dir, 'retrained.model'))
	with open(map_filename, 'r') as f:
		with open(os.path.join(output_dir, 'predictions.csv'), 'w') as g:
			g.write('filename,true_label,pred_label\n')
			for line in f:
				filename, true_ind = line.strip().split('\t')
				image_data = np.array(Image.open(filename), dtype=np.float32)
				image_data = np.ascontiguousarray(np.transpose(
					image_data[:, :, ::-1], (2,0,1)))
				dnn_output = loaded_model.eval(
					{loaded_model.arguments[0]: [image_data]})
				true_label = inds_to_labels[int(true_ind)]
				pred_label = inds_to_labels[np.argmax(np.squeeze(dnn_output))]
				g.write('{},{},{}\n'.format(filename, true_label, pred_label))

	df = pd.read_csv(os.path.join(output_dir, 'predictions.csv'))
	num_correct = len(df.loc[df['true_label'] == df['pred_label']].index)
	print('Overall accuracy on test set: {:0.3f}'.format(
		  len(df.loc[df['true_label'] == df['pred_label']].index) /
		  len(df.index)))

	return


def main(input_dir, validation_dir, output_dir, model_filename, num_epochs,
	model_type, retraining_type):
	''' Coordinates all activities for the script '''

	# Create a temporary directory to house the MAP file
	with tempfile.TemporaryDirectory() as temp_dir:
		training_map_filename = os.path.join(temp_dir, 'map_train.tsv')
		validation_map_filename = os.path.join(temp_dir, 'map_test.tsv')
		
		_, _ = write_map_file(validation_map_filename, input_dir, output_dir)
		num_classes, epoch_size = write_map_file(training_map_filename,
												 input_dir, output_dir)

		my_rank = retrain_model(training_map_filename, output_dir,
								num_classes, epoch_size, model_filename,
								num_epochs, model_type, retraining_type)
		if my_rank == 0:
			evaluate_model(validation_map_filename, output_dir, num_classes)

	return
    

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''
Retrains a pretrained DNN model using supplied images. Creates MAP files
(in a temporary directory) used by ImageDeserializer during training and
validation. Outputs the retrained model, a tsv file mapping the class names to
indices, and the validation set predictions to the specified directory.
''')
	parser.add_argument('-i', '--input_dir', type=str, required=True,
						help='Directory containing all training image files' +
						' in subfolders named by class.')
	parser.add_argument('-v', '--validation_dir', type=str, required=True,
						help='Directory containing all test image files' +
						' in subfolders named by class.')
	parser.add_argument('-o', '--output_dir',
						type=str, required=True,
						help='Output directory for the model. Supporting ' +
						     'files will be placed in the same folder.')
	parser.add_argument('-m', '--model_filename',
						type=str, required=True,
						help='Filepath of the pretrained model.')
	parser.add_argument('-n', '--num_epochs',
						type=int, required=True,
						help='Number of epochs to retrain the model.')
	parser.add_argument('-t', '--model_type', type=str, required=True,
						help='The model type to retrain, which should be ' +
						'either "resnet18" or "alexnet".')
	parser.add_argument('-r', '--retraining_type',
						type=str, required=True,
						help='Specifies which layers to retrain in the model.' +
						' Should be one of "last_only", "fully_connected", ' +
						'or "all". Cannot use "fully_connected" retraining ' +
						'type with "resnet18" model type.')
	args = parser.parse_args()

	# Ensure argument values are acceptable before proceeding
	assert os.path.exists(args.input_dir), \
		'Input directory {} does not exist'.format(args.input_dir)
	assert os.path.exists(args.validation_dir), \
		'Validation directory {} does not exist'.format(args.validation_dir)
	assert os.path.exists(args.model_filename), \
		'Model file {} does not exist'.format(args.model_filename)
	assert args.num_epochs > 0, 'Number of epochs must be greater than zero'
	assert args.model_type in ['resnet18', 'alexnet'], \
		'Model type must be "resnet18" or "alexnet" (without the quotes).'
	assert args.retraining_type in ['last_only', 'fully_connected', 'all'], \
		'Retraining type must be "last_only", "fully_connected", or "all" ' + \
		'(without the quotes).'
	if (args.retraining_type == 'fully_connected') and \
		(args.model_type == 'resnet18'):
		raise Exception('Can only use "all" or "last_only" retraining types ' +
						'with ResNet 18.')
	os.makedirs(args.output_dir, exist_ok=True)

	main(args.input_dir, args.validation_dir, args.output_dir,
		 args.model_filename, args.num_epochs, args.model_type,
		 args.retraining_type)