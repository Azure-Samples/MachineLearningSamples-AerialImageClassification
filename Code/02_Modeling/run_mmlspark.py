'''
run_mmlspark.py
(c) Microsoft Corporation, 2017

Trains an MMLSpark model to classify images featurized by a specified CNTK
pretrained model. Saves the model and test set predictions to blob storage.
Logs some evaluation metrics directly to run history.
'''

import os, time, mmlspark, pyspark, argparse
import numpy as np
from io import BytesIO
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.classification import RandomForestClassifier, \
	LogisticRegression
from azureml.logging import get_azureml_logger
import pandas as pd
from configparser import ConfigParser
from azure.storage.blob import BlockBlobService


def ensure_str(str_data):
	''' Helper function to correct type of imported strings '''
	if isinstance(str_data, str):
		return(str_data)
	return(str_data.encode('utf-8'))

class ConfigFile(object):
	''' Copies ConfigParser results into attributes, correcting type '''
	def __init__(self, config_filename, pretrained_model_type,
		mmlspark_model_type, output_model_name):
		''' Load static info for cluster/job creation from a config file '''
		config = ConfigParser(allow_no_value=True)
		config.read(config_filename)
		my_config = config['Settings']
		self.spark = pyspark.sql.SparkSession.builder.appName('vienna') \
			.getOrCreate()

		self.pretrained_model_type = pretrained_model_type
		self.mmlspark_model_type = mmlspark_model_type
		self.output_model_name = output_model_name

		# Storage account where results will be written
		self.storage_account_name = ensure_str(
			my_config['storage_account_name'])
		self.storage_account_key = ensure_str(my_config['storage_account_key'])
		self.container_pretrained_models = ensure_str(
			my_config['container_pretrained_models'])
		self.container_trained_models = ensure_str(
			my_config['container_trained_models'])
		self.container_data_training = ensure_str(
			my_config['container_data_training'])
		self.container_data_testing = ensure_str(
			my_config['container_data_testing'])
		self.container_prediction_results = ensure_str(
			my_config['container_prediction_results'])

		# URIs where data will be loaded or saved
		self.train_uri = 'wasb://{}@{}.blob.core.windows.net/*/*.png'.format(
			self.container_data_training, self.storage_account_name)
		self.test_uri = 'wasb://{}@{}.blob.core.windows.net/*/*.png'.format(
			self.container_data_testing, self.storage_account_name)
		self.model_uri = 'wasb://{}@{}.blob.core.windows.net/{}'.format(
			self.container_pretrained_models, self.storage_account_name,
			'ResNet_18.model' if pretrained_model_type == 'resnet18' \
			else 'AlexNet.model')
		self.output_uri = 'wasb://{}@{}.blob.core.windows.net/{}/model'.format(
			self.container_trained_models, self.storage_account_name,
			output_model_name)
		self.predictions_filename = '{}_predictions_test_set.csv'.format(
			output_model_name)

		# Load the pretrained model
		self.last_layer_name = 'z.x' if (pretrained_model_type == 'resnet18') \
			else 'h2_d'
		self.cntk_model = mmlspark.CNTKModel().setInputCol('unrolled') \
			.setOutputCol('features') \
			.setModelLocation(self.spark, self.model_uri) \
			.setOutputNodeName(self.last_layer_name)

		# Initialize other Spark pipeline components
		self.extract_label_udf = udf(lambda row: os.path.basename(
										os.path.dirname(row.path)),
									 StringType())
		self.extract_path_udf = udf(lambda row: row.path, StringType())
		if mmlspark_model_type == 'randomforest':
			self.mmlspark_model_type = RandomForestClassifier(numTrees=20,
															  maxDepth=5)
		elif mmlspark_model_type == 'logisticregression':
			self.mmlspark_model_type = LogisticRegression(regParam=0.01,
														  maxIter=10)
		self.unroller = mmlspark.UnrollImage().setInputCol('image') \
			.setOutputCol('unrolled')

		return


def write_model_summary_to_blob(config, mmlspark_model_type):
	''' Writes a summary file describing the model to be used during o16n '''
	output_str = '''output_model_name,{}
model_source,mmlspark
pretrained_model_type,{}
retraining_type,last_only
mmlspark_model_type,{}
'''.format(config.output_model_name, config.pretrained_model_type,
		   mmlspark_model_type)
	file_name = '{}/model.info'.format(config.output_model_name)
	blob_service = BlockBlobService(config.storage_account_name,
									config.storage_account_key)
	blob_service.create_container(config.container_trained_models)
	blob_service.create_blob_from_text(
			config.container_trained_models, file_name, output_str)
	return


def load_data(data_uri, config, sample_frac):
	df = config.spark.readImages(data_uri, recursive=True,
		sampleRatio=sample_frac).toDF('image')
	df = df.withColumn('label', config.extract_label_udf(df['image']))
	df = df.withColumn('filepath', config.extract_path_udf(df['image']))
	df = config.unroller.transform(df).select('filepath', 'unrolled', 'label')
	df = config.cntk_model.transform(df).select(
		['filepath', 'features', 'label'])
	return(df)


def main(pretrained_model_type, mmlspark_model_type, config_filename,
		 output_model_name, sample_frac):
	# Load the configuration file
	config = ConfigFile(config_filename, pretrained_model_type,
		mmlspark_model_type, output_model_name)
	write_model_summary_to_blob(config, mmlspark_model_type)

	# Log the parameters of the run
	run_logger = get_azureml_logger()
	run_logger.log('pretrained_model_type', pretrained_model_type)
	run_logger.log('mmlspark_model_type', mmlspark_model_type)
	run_logger.log('config_filename', config_filename)
	run_logger.log('output_model_name', output_model_name)
	run_logger.log('sample_frac', sample_frac)

	# Train and save the MMLSpark model
	train_df = load_data(config.train_uri, config, sample_frac)
	mmlspark_model = mmlspark.TrainClassifier(
		model=config.mmlspark_model_type, labelCol='label').fit(train_df)
	mmlspark_model.write().overwrite().save(config.output_uri)

	# Apply the MMLSpark model to the test set and save the accuracy metric
	test_df = load_data(config.test_uri, config, sample_frac)
	predictions = mmlspark_model.transform(test_df)
	metrics = mmlspark.ComputeModelStatistics(evaluationMetric='accuracy') \
		.transform(predictions)
	metrics.show()
	run_logger.log('accuracy_on_test_set', metrics.first()['accuracy'])
	
	# Save the predictions
	tf = mmlspark.IndexToValue().setInputCol('scored_labels') \
		.setOutputCol('pred_label')
	predictions = tf.transform(predictions).select(
		'filepath', 'label', 'pred_label')
	output_str = predictions.toPandas().to_csv(index=False)
	blob_service = BlockBlobService(config.storage_account_name,
									config.storage_account_key)
	blob_service.create_container(config.container_prediction_results)
	blob_service.create_blob_from_text(
			config.container_prediction_results,
			config.predictions_filename,
			output_str)

	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''
Trains an MMLSpark model to classify images featurized by a specified CNTK
pretrained model. Saves the model and test set predictions to blob storage.
Logs some evaluation metrics directly to run history.''')
	parser.add_argument('-p', '--pretrained_model_type', type=str,
						required=True,
						help='The model type to retrain, which should be ' +
						'either "resnet18" or "alexnet".')
	parser.add_argument('-m', '--mmlspark_model_type',
						type=str, required=True,
						help='Specifies which type of model should be ' +
						'trained on featurized images. Should be either ' +
						'"randomforest" or "logisticregresssion".')
	parser.add_argument('-c', '--config_filename',
						type=str, required=True,
						help='Filepath of the configuration file specifying ' +
						'credentials for a storage account, container ' +
						'registry, and Batch AI training itself.')
	parser.add_argument('-o', '--output_model_name',
						type=str, required=True,
						help='Retrained model files will be saved under this ' +
						'"subdirectory" (prefix) in the trained model blob ' +
						'container specified by the config file.')
	parser.add_argument('-f', '--sample_frac',
						type=float, required=False, default=1.0,
						help='Subsamples training and test data for faster ' +
						'results. Default sampling fraction is 1.0 (all ' +
						'samples used).')
	args = parser.parse_args()

	assert args.pretrained_model_type in ['resnet18', 'alexnet'], \
		'Pretrained model type must be "resnet18" or "alexnet".'
	assert args.mmlspark_model_type in ['randomforest', 'logisticregression'], \
		'MMLSpark model type must be "randomforest" or "logisticregression".'
	assert os.path.exists(args.config_filename), \
		'Could not find config file {}'.format(args.config_filename)
	assert (args.sample_frac <= 1.0) and (args.sample_frac > 0.0), \
		'Sampling fraction must be between 0.0 and 1.0.'

	print('Arguments ok...preparing to run')
	main(args.pretrained_model_type, args.mmlspark_model_type,
		 args.config_filename, args.output_model_name, args.sample_frac)