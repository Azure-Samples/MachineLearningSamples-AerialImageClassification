'''
batch_score_spark.py
by Mary Wahl
(c) Microsoft Corporation, 2017

Applies a trained MMLSpark model to a large static dataset in an HDInsight
cluster's associated blob storage account. This script require the following
arguments:
- config_filename:   Includes storage account credentials and container names
- output_model_name: The model name specified at the time of training; used for
                     lookup of output files in blob storage.
'''
import os, io, argparse, mmlspark, pyspark
from azureml.logging import get_azureml_logger
import numpy as np
import pandas as pd
from configparser import ConfigParser
from azure.storage.blob import BlockBlobService
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.feature import IndexToString
from mmlspark import TrainedClassifierModel

def ensure_str(str_data):
	''' Helper function to correct type of imported strings '''
	if isinstance(str_data, str):
		return(str_data)
	return(str_data.encode('utf-8'))

class ConfigFile(object):
	''' Copies ConfigParser results into attributes, correcting type '''
	def __init__(self, config_filename, output_model_name):
		''' Load/validate model information from a config file '''
		config = ConfigParser(allow_no_value=True)
		config.read(config_filename)
		my_config = config['Settings']
		self.spark = pyspark.sql.SparkSession.builder.appName('vienna') \
			.getOrCreate()

		# Load storage account info
		self.storage_account_name = ensure_str(
			my_config['storage_account_name'])
		self.storage_account_key = ensure_str(my_config['storage_account_key'])
		self.container_pretrained_models = ensure_str(
			my_config['container_pretrained_models'])
		self.container_trained_models = ensure_str(
			my_config['container_trained_models'])
		self.container_data_o16n = ensure_str(
			my_config['container_data_o16n'])
		self.container_prediction_results = ensure_str(
			my_config['container_prediction_results'])
		self.predictions_filename = '{}_predictions_o16n.csv'.format(
			output_model_name)

		# Load blob service and ensure containers are available
		blob_service = BlockBlobService(self.storage_account_name,
										self.storage_account_key)
		container_list = [i.name for i in blob_service.list_containers()]
		for container in [self.container_pretrained_models,
						  self.container_trained_models,
						  self.container_data_o16n,
						  self.container_prediction_results]:
			assert container in container_list, \
				'Could not find container {} in storage '.format(container) + \
				'account {}'.format(self.storage_account_name)

		# Load information on the named model
		self.output_model_name = output_model_name
		description = blob_service.get_blob_to_text(
			container_name=self.container_trained_models,
			blob_name='{}/model.info'.format(self.output_model_name))
		description_dict = {}
		for line in description.content.split('\n'):
			if len(line) == 0:
				continue
			key, val = line.strip().split(',')
			description_dict[key] = val
		self.model_source = description_dict['model_source']
		self.pretrained_model_type = description_dict['pretrained_model_type']

		# Create pipeline components common to both model types
		self.extract_path_udf = udf(lambda row: os.path.basename(row.path),
									StringType())
		self.unroller = mmlspark.UnrollImage().setInputCol('image') \
			.setOutputCol('unrolled')
		return


def load_mmlspark_model_components(config):
	''' Loads all components needed to apply a trained MMLSpark model '''
	# Load the pretrained featurization model
	if config.pretrained_model_type == 'resnet18':
		model_filename = 'ResNet_18.model'
		last_layer_name = 'z.x'
	elif config.pretrained_model_type == 'alexnet':
		model_filename = 'AlexNet.model'
		last_layer_name = 'h2_d'
	model_uri = 'wasb://{}@'.format(config.container_pretrained_models) + \
				'{}.blob.core.windows'.format(config.storage_account_name) + \
				'.net/{}'.format(model_filename)
	config.cntk_model = mmlspark.CNTKModel().setInputCol('unrolled') \
		.setOutputCol('features').setModelLocation(config.spark, model_uri) \
		.setOutputNodeName(last_layer_name)

	# Load the MMLSpark-trained model
	mmlspark_uri = 'wasb://{}@'.format(config.container_trained_models) + \
				   '{}.blob.core.'.format(config.storage_account_name) + \
				   'windows.net/{}/model'.format(config.output_model_name)
	config.mmlspark_model = TrainedClassifierModel.load(mmlspark_uri)

	# Load the transform that will convert model output from indices to strings
	config.tf = mmlspark.IndexToValue().setInputCol('scored_labels') \
		.setOutputCol('pred_label')

	return(config)


def load_data(config, sample_frac=1.0):
	data_uri = 'wasb://{}@{}.blob.core.windows.net/*.png'.format(
			config.container_data_o16n, config.storage_account_name)
	df = config.spark.readImages(data_uri, recursive=True,
		sampleRatio=sample_frac).toDF('image')
	df = df.withColumn('filepath', config.extract_path_udf(df['image']))
	df = config.unroller.transform(df).select('filepath', 'unrolled')
	df = config.cntk_model.transform(df).select(
		['filepath', 'features'])
	return(df)


def main(config_filename, output_model_name, sample_frac):
	''' Coordinate application of trained models to large static image set '''
	config = ConfigFile(config_filename, output_model_name)

	if config.model_source == 'mmlspark':
		config = load_mmlspark_model_components(config)
	else:
		raise Exception('Model source not recognized')

	df = load_data(config, sample_frac)

	predictions = config.mmlspark_model.transform(df)
	predictions = config.tf.transform(predictions).select(
		'filepath', 'pred_label')

	output_str = predictions.toPandas().to_csv(index=False)
	blob_service = BlockBlobService(config.storage_account_name,
									config.storage_account_key)
	blob_service.create_blob_from_text(
			config.container_prediction_results,
			config.predictions_filename,
			output_str)

	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''
Applies a trained MMLSpark model to a large static dataset in an HDInsight
cluster's associated blob storage account.
''')
	parser.add_argument('-c', '--config_filename',
						type=str, required=True,
						help='Includes storage account credentials and ' +
						'container names.')
	parser.add_argument('-o', '--output_model_name',
						type=str, required=True,
						help='The model name specified at the time of ' + \
						'training; used for lookup of output files in ' + \
						'blob storage.')
	parser.add_argument('-f', '--sample_frac',
						type=float, required=False, default=1.0,
						help='Subsamples data. Default sampling fraction is ' +
						'1.0 (all samples used).')
	args = parser.parse_args()

	assert os.path.exists(args.config_filename), \
		'Could not find config file {}'.format(args.config_filename)
	main(args.config_filename, args.output_model_name, args.sample_frac)