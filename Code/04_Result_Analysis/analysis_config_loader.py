'''
analysis_config_loader.py
by Mary Wahl
(c) Microsoft Corporation, 2017

Loads dataframes of prediction results and the description of a trained model.
'''
import os, io
import numpy as np
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
	def __init__(self, config_filename, output_model_name):
		''' Load/validate model information from a config file '''
		config = ConfigParser(allow_no_value=True)
		config.read(config_filename)
		my_config = config['Settings']
		self.output_model_name = output_model_name

		# Load storage account info
		self.storage_account_name = ensure_str(
			my_config['storage_account_name'])
		self.storage_account_key = ensure_str(my_config['storage_account_key'])
		self.container_prediction_results = ensure_str(
			my_config['container_prediction_results'])
		self.container_trained_models = ensure_str(
			my_config['container_trained_models'])
		self.container_data_o16n = ensure_str(
			my_config['container_data_o16n'])
		self.predictions_o16n_filename = '{}_predictions_o16n.csv'.format(
			output_model_name)
		self.predictions_test_filename = '{}_predictions_test_set.csv'.format(
			output_model_name)

		# Load blob service and ensure containers are available
		blob_service = BlockBlobService(self.storage_account_name,
										self.storage_account_key)
		container_list = [i.name for i in blob_service.list_containers()]
		for container in [self.container_trained_models,
						  self.container_prediction_results,
						  self.container_data_o16n]:
			assert container in container_list, \
				'Could not find container {} in storage '.format(container) + \
				'account {}'.format(self.storage_account_name)

		# Load the predictions themselves
		try:
			o16n_blob = blob_service.get_blob_to_text(
				container_name=self.container_prediction_results,
				blob_name=self.predictions_o16n_filename)
			self.o16n_df = pd.read_csv(io.StringIO(o16n_blob.content))
		except Exception as e:
			raise Exception('Error loading operationalization predictions;' +
				'did you run batch_score_spark.py with this model?\n{}'.format(
					e))
		self.o16n_df['name'] = self.o16n_df['filepath'].apply(
			lambda x: os.path.basename(x))
		self.o16n_df.drop('filepath', axis=1, inplace=True)

		try:
			test_blob = blob_service.get_blob_to_text(
				container_name=self.container_prediction_results,
				blob_name=self.predictions_test_filename)
			self.test_df = pd.read_csv(io.StringIO(test_blob.content))
		except Exception as e:
			raise Exception('Error downloading test set predictions:' +
				'\n{}'.format(e))

		try:
			tile_blob = blob_service.get_blob_to_text(
				container_name=self.container_data_o16n,
				blob_name='tile_summaries.csv')
			self.tile_summaries_df = pd.read_csv(io.StringIO(tile_blob.content))
		except Exception as e:
			raise Exception('Error downloading tile summaries for o16n data:' +
				'\n{}'.format(e))
		self.tile_summaries_df['name'] = self.tile_summaries_df['filename'] \
			.apply(lambda x: os.path.basename(x))
		self.tile_summaries_df.drop('filename', axis=1, inplace=True)
		self.o16n_df = self.o16n_df.merge(self.tile_summaries_df,
										   on='name', how='inner')
		self.o16n_df = self.o16n_df[['name', 'pred_label', 'llcrnrlat',
			'llcrnrlon', 'urcrnrlat', 'urcrnrlon']]

		# Load the description of the trained model
		try:
			description = blob_service.get_blob_to_text(
				container_name=self.container_trained_models,
				blob_name='{}/model.info'.format(self.output_model_name))
		except Exception as e:
			raise Exception('Error downloading model description:' +
				'\n{}'.format(e))
		description_dict = {}
		for line in description.content.split('\n'):
			if len(line) == 0:
				continue
			key, val = line.strip().split(',')
			description_dict[key] = val
		self.model_source = description_dict['model_source']
		self.pretrained_model_type = description_dict['pretrained_model_type']
		self.mmlspark_model_type = description_dict['mmlspark_model_type']
		return
