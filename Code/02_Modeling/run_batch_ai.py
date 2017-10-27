'''
run_batch_ai.py
(c) Microsoft Corporation, 2017

This script is designed to call Batch AI training from Vienna and log the
results to Vienna's run history feature. This script assumes that the
associated config file and Azure file share have been set up in advance. It
waits for the cluster to reach steady state (if necessary), submits the job,
downloads its output after completion, and finally parses the output
files to return metrics to Vienna's run history.
'''
import argparse, os, time, datetime, requests, re
import azure.mgmt.batchai as training
import azure.mgmt.batchai.models as tm
from azure.common.credentials import ServicePrincipalCredentials
from azureml.logging import get_azureml_logger
import pandas as pd
from configparser import ConfigParser
from azure.storage.file import FileService
from azure.storage.blob import BlockBlobService
from tempfile import TemporaryFile

def ensure_str(str_data):
	''' Helper function to correct type of imported strings '''
	if isinstance(str_data, str):
		return(str_data)
	return(str_data.encode('utf-8'))

class ConfigFile(object):
	''' Copies ConfigParser results into attributes, correcting type '''
	def __init__(self, config_filename):
		''' Load static info for cluster/job creation from a config file '''
		config = ConfigParser(allow_no_value=True)
		config.read(config_filename)
		my_config = config['Settings']

		# General info needed for creating clients/clusters/jobs
		self.bait_subscription_id = ensure_str(my_config['bait_subscription_id'])
		self.bait_aad_client_id = ensure_str(my_config['bait_aad_client_id'])
		self.bait_aad_secret = ensure_str(my_config['bait_aad_secret'])
		self.bait_aad_token_uri = 'https://login.microsoftonline.com/' + \
			'{0}/oauth2/token'.format(ensure_str(my_config['bait_aad_tenant']))
		self.bait_region = ensure_str(my_config['bait_region'])
		self.bait_resource_group_name = ensure_str(
			my_config['bait_resource_group_name'])
		self.bait_vms_in_cluster = int(my_config['bait_vms_in_cluster'])
		self.bait_vms_per_job = int(my_config['bait_vms_per_job'])
		self.bait_cluster_name = ensure_str(my_config['bait_cluster_name'])

		assert self.bait_vms_per_job <= self.bait_vms_in_cluster, \
			'Number of VMs in cluster ({}) < Number of VMs for job ({}'.format(
				self.bait_vms_in_cluster, self.bait_vms_in_cluster)
		assert self.bait_vms_in_cluster > 0, \
			'Number of VMs used for the job must be greater than zero.'

		# Storage account where results will be written
		self.storage_account_name = ensure_str(
			my_config['storage_account_name'])
		self.storage_account_key = ensure_str(my_config['storage_account_key'])
		self.storage_account_fileshare_url = 'https://' + \
			'{}.file.core.windows.net/baitshare'.format(
				self.storage_account_name)
		self.container_trained_models = ensure_str(
			my_config['container_trained_models'])
		self.predictions_container = ensure_str(
			my_config['container_prediction_results'])

		return


def write_model_summary_to_blob(config, output_model_name,
	pretrained_model_type, retraining_type):
	''' Writes a summary file describing the model to be used during o16n '''
	output_str = '''output_model_name,{}
model_source,batchaitraining
pretrained_model_type,{}
retraining_type,{}
mmlspark_model_type,none
'''.format(output_model_name, pretrained_model_type, retraining_type)
	file_name = '{}/model.info'.format(output_model_name)
	blob_service = BlockBlobService(config.storage_account_name,
									config.storage_account_key)
	blob_service.create_container(config.container_trained_models)
	blob_service.create_blob_from_text(
			config.container_trained_models, file_name, output_str)
	return


def get_client(config):
	''' Connect to Batch AI '''
	client = training.BatchAIManagementClient(
		credentials=ServicePrincipalCredentials(
			client_id=config.bait_aad_client_id,
			secret=config.bait_aad_secret,
			token_uri=config.bait_aad_token_uri),
		subscription_id=config.bait_subscription_id,
		base_url=None)
	return(client)


def get_cluster(config):
	'''
	Checks whether a cluster with the specified name already exists. If so, it
	uses that cluster; otherwise, it creates a new one.
	'''
	client = get_client(config)

	# Start cluster creation if necessary
	try:
		cluster = client.clusters.get(config.bait_resource_group_name,
										 config.bait_cluster_name)
	except:
		print('Error: could not find cluster named {}'.format(
			config.bait_cluster_name))

	return(cluster)


def check_for_steady_cluster_status(config, max_sec_to_wait=1200):
	'''
	Waits until the cluster reaches a "steady" status. Checks every ten
	seconds.
	'''
	client = get_client(config)
	start = time.time()
	while (time.time() - start < max_sec_to_wait):
		cluster = client.clusters.get(config.bait_resource_group_name,
									 config.bait_cluster_name)
		if cluster.allocation_state == tm.AllocationState.steady:
			print('Cluster has reached "steady" allocation state. Ready for ' +
				  'job submission.')
			if cluster.errors is not None:
				raise Exception('Errors were thrown during cluster creation:' +
					            '\n{}'.format('\n'.join(cluster.errors)))
			return
		time.sleep(10)
	raise Exception('Max wait time exceeded for cluster to reach "steady" ' +
					'state ({} seconds).'.format(max_sec_to_wait))

	
def submit_job(config, pretrained_model_type, retraining_type,
			   output_model_name, num_epochs):
	''' Defines and submits a job. Does not check for completion. '''
	client = get_client(config)
	job_name = 'job{}'.format(
		datetime.datetime.utcnow().strftime('%m_%d_%H_%M_%S'))
	cluster = client.clusters.get(config.bait_resource_group_name,
								 config.bait_cluster_name)

	# Define the command line arguments to the retraining script
	command_line_args = '--input_dir $AZ_BATCHAI_INPUT_TRAININGDATA ' + \
		'--validation_dir $AZ_BATCHAI_INPUT_VALIDATIONDATA ' + \
		'--output_dir $AZ_BATCHAI_OUTPUT_MODEL ' + \
		'--num_epochs {} '.format(num_epochs) + \
		'--retraining_type {} '.format(retraining_type) + \
		'--model_type {} '.format(pretrained_model_type) + \
		'--model_filename $AZ_BATCHAI_INPUT_PRETRAINEDMODELS/'
	if pretrained_model_type == 'alexnet':
		command_line_args += 'AlexNet.model'
	elif pretrained_model_type == 'resnet18':
		command_line_args += 'ResNet_18.model'

	# Define the job
	cntk_settings = tm.CNTKsettings(
		language_type='python',
		python_script_file_path='$AZ_BATCHAI_INPUT_SCRIPT/' +
			'retrain_model_distributed.py',
		command_line_args=command_line_args,
		process_count=config.bait_vms_per_job) # NC6s -- one GPU per VM

	job_create_params = tm.job_create_parameters.JobCreateParameters(
		location=config.bait_region,
		cluster=tm.ResourceId(cluster.id),                
		node_count=config.bait_vms_per_job,
		std_out_err_path_prefix='$AZ_BATCHAI_MOUNT_ROOT/afs', 
		output_directories=[
			tm.OutputDirectory(
				id='MODEL',
				path_prefix='$AZ_BATCHAI_MOUNT_ROOT/afs')],
		input_directories=[
			tm.InputDirectory(
				id='SCRIPT',
				path='$AZ_BATCHAI_MOUNT_ROOT/afs/scripts'),
			tm.InputDirectory(
				id='PRETRAINEDMODELS',
				path='$AZ_BATCHAI_MOUNT_ROOT/afs/pretrainedmodels'),
			tm.InputDirectory(
				id='TRAININGDATA',
				path='$AZ_BATCHAI_MOUNT_ROOT/nfs/training_images'),
			tm.InputDirectory(
				id='VALIDATIONDATA',
				path='$AZ_BATCHAI_MOUNT_ROOT/nfs/validation_images')],
        cntk_settings=cntk_settings)

	# Submit the job
	job = client.jobs.create(
		resource_group_name=config.bait_resource_group_name,
		job_name=job_name,
		parameters=job_create_params)    

	return(job_name)


def check_for_job_completion(config, job_name, max_sec_to_wait=7200):
	''' Check for the job status to change indicating completion '''
	client = get_client(config)
	time.sleep(10)
	start = time.time()
	while (time.time() - start < max_sec_to_wait):
		job = client.jobs.get(config.bait_resource_group_name, job_name)
		if (job.execution_state == tm.ExecutionState.succeeded) or \
			(job.execution_state == tm.ExecutionState.failed):
			return
		time.sleep(10)
	raise Exception('Max wait time exceeded for job completion ' +
					'({} seconds).'.format(max_sec_to_wait))


def download_from_file_share(azure_filename, local_filename):
	''' Save an output file from Azure File Share '''
	r = requests.get(azure_filename, stream=True)
	with open(local_filename, 'wb') as f:
		for chunk in r.iter_content(chunk_size=512 * 1024):
			if chunk:
				f.write(chunk)


def transfer_fileshare_to_blob(config, fileshare_uri, output_model_name):
	''' NB -- transfer proceeds via local temporary file! '''
	file_service = FileService(config.storage_account_name,
							   config.storage_account_key)
	blob_service = BlockBlobService(config.storage_account_name,
									config.storage_account_key)
	blob_service.create_container(config.container_trained_models)
	blob_service.create_container(config.predictions_container)

	uri_core = fileshare_uri.split('.file.core.windows.net/')[1].split('?')[0]
	fields = uri_core.split('/')
	fileshare = fields.pop(0)
	subdirectory = '/'.join(fields[:-1])
	file_name = '{}/{}'.format(output_model_name, fields[-1])
	
	with TemporaryFile() as f:
		file_service.get_file_to_stream(share_name=fileshare,
										directory_name=subdirectory,
										file_name=fields[-1],
										stream=f)
		f.seek(0)
		if 'predictions' in fields[-1]:
			blob_service.create_blob_from_stream(
				config.predictions_container,
				'{}_predictions_test_set.csv'.format(output_model_name),
				f)
		else:
			blob_service.create_blob_from_stream(
				config.container_trained_models, file_name, f)

	return


def retrieve_outputs(config, job_name, output_model_name):
	''' Get stdout, stderr, retrained model, and label-to-index dict '''
	client = get_client(config)
	status_files = client.jobs.list_output_files(
		resource_group_name=config.bait_resource_group_name,
		job_name=job_name,
		jobs_list_output_files_options=tm.JobsListOutputFilesOptions('stdOuterr'))
	for file in list(status_files):
		download_from_file_share(file.download_url,
								 os.path.join('outputs', file.name))

	output_files = client.jobs.list_output_files(
		resource_group_name=config.bait_resource_group_name,
		job_name=job_name,
		jobs_list_output_files_options=tm.JobsListOutputFilesOptions('MODEL'))
	for file in list(output_files):
		transfer_fileshare_to_blob(config, file.download_url, output_model_name)

	client.jobs.delete(resource_group_name=config.bait_resource_group_name,
					   job_name=job_name)
	return


def parse_stdout(run_logger):
	''' Parse the training logs and record using Vienna SDK '''
	with open(os.path.join('outputs', 'stdout.txt'), 'r') as f:
		lines = f.readlines()

	progress_re = 'Finished Epoch\[(\d+) of \d+\]: \[Training\] loss = ' + \
			      '([0-9.]+) \* [0-9]+, metric = ([0-9.]+)% \* [0-9]+ ' + \
			      '([0-9.]+)s \( ([0-9.]+) samples/s\);'
	progress_re2 = 'Finished Epoch\[(\d+) of \d+\]: \[Training\] loss = ' + \
			       '([0-9.]+) \* [0-9]+, metric = ([0-9.]+)% \* [0-9]+ ' + \
			       '([0-9.]+)s \(([0-9.]+) samples/s\);'
	p = re.compile(progress_re)
	p2 = re.compile(progress_re2)

	progress_lines = []
	for line in lines:
		m = p.match(line)
		if m is not None:
			progress_lines.append(list(m.groups()))
		else: # try a minor variation
			m = p2.match(line)
			if m is not None:
				progress_lines.append(list(m.groups()))

	df = pd.DataFrame(progress_lines,
					  columns=['epoch', 'loss', 'accuracy', 'duration', 'rate'],
					  dtype=float).groupby('epoch').mean().reset_index()
	run_logger.log('training_loss', df['loss'].values.tolist())	
	run_logger.log('training_error_pct', df['accuracy'].values.tolist())
	run_logger.log('epoch_duration', df['duration'].values.tolist())
	run_logger.log('samples_per_sec', df['rate'].values.tolist())

	accuracy_re = 'Overall accuracy on test set: ([0-9.]+)'
	p = re.compile(accuracy_re)
	for line in lines:
		m = p.match(line)
		if m is not None:
			print('Test set accuracy: {}'.format(m.groups(1)[0]))
			run_logger.log('test_set_accuracy', m.groups(1)[0])
	return

def main(pretrained_model_type, retraining_type, config_filename,
		 output_model_name, num_epochs):
	''' Coordinate all activities for Batch AI training '''

	# Log the parameters used for this run
	run_logger = get_azureml_logger()
	run_logger.log('pretrained_model_type', pretrained_model_type)
	run_logger.log('config_filename', config_filename)
	run_logger.log('retraining_type', retraining_type)
	run_logger.log('output_model_name', output_model_name)

	# Load the configuration file and save relevant info
	config = ConfigFile(config_filename)
	write_model_summary_to_blob(config, output_model_name,
		pretrained_model_type, retraining_type)

	# Create a cluster (if necessary) and wait till it's ready
	get_cluster(config)
	check_for_steady_cluster_status(config)

	# Submit the job and wait until it completes
	job_name = submit_job(config, pretrained_model_type, retraining_type,
						  output_model_name, num_epochs)
	print('Job submitted: checking for job completion')
	check_for_job_completion(config, job_name)
	print('Job complete: retrieving output files')

	# Download the output files and store metrics to Vienna
	retrieve_outputs(config, job_name, output_model_name)
	print('Parsing output logs')
	parse_stdout(run_logger)

	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''
Orchestrates pretrained image classifier retraining through Batch AI training.
Can retrain multiple model types and to different depths. The training data for
this example is fixed and provided in the docker image specified in the config
file.
''')
	parser.add_argument('-p', '--pretrained_model_type', type=str,
						required=True,
						help='The model type to retrain, which should be ' +
						'either "resnet18" or "alexnet".')
	parser.add_argument('-r', '--retraining_type',
						type=str, required=True,
						help='Specifies which layers to retrain in the model.' +
						' Should be one of "last_only", "fully_connected", ' +
						'or "all".')
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
						help='Subsamples data. Default sampling fraction is ' +
						'1.0 (all samples used).')
	parser.add_argument('-n', '--num_epochs',
						type=int, required=False, default=10,
						help='Number of epochs to retrain the model for.')
	args = parser.parse_args()

	# Ensure specified files/directories exist
	assert args.pretrained_model_type in ['resnet18', 'alexnet'], \
		'Pretrained model type must be "resnet18" or "alexnet".'
	assert args.retraining_type in ['last_only', 'fully_connected', 'all'], \
		'Retraining type must be "last_only", "fully_connected", or "all" ' + \
		'(without the quotes).'
	assert os.path.exists(args.config_filename), \
		'Could not find config file {}'.format(args.config_filename)
	assert args.num_epochs > 0, 'Number of epochs must be greater than zero'
	os.makedirs('outputs', exist_ok=True)

	main(args.pretrained_model_type, args.retraining_type, args.config_filename,
		 args.output_model_name, args.num_epochs)