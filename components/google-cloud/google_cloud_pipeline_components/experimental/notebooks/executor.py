# In the class, add a method to create the component.yaml
# Make component.py a factory in my class so I have only one file.

#!/usr/bin/env python
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Source code used by the KFP component for running notebooks as a step.

Uses the Notebooks Executor API:
https://cloud.google.com/notebooks/docs/reference/rest#rest-resource:-v1.projects.locations.executions.
"""

from googleapiclient import discovery
from googleapiclient import errors
from googleapiclient.http import set_user_agent
import httplib2
from kfp.components import create_component_from_func, InputPath, OutputPath
from kfp.v2.dsl import component
import time
from types import SimpleNamespace
from typing import NamedTuple, Optional  # pylint: disable=unused-import

class NotebooksExecutor:
  """Component logic for executing notebooks as a step."""

  @staticmethod
  @component(
      base_image='python:3.8',
      packages_to_install=['google-api-python-client'],
      output_component_file='component.yaml',
  )
  def create_component_file(
      project_id: str,
      input_notebook_file: str,
      output_notebook_folder: str,
      execution_id: str,
      location: str = 'us-central1',
      master_type: str = 'n1-standard-4',
      scale_tier: str = 'CUSTOM',
      accelerator_type: Optional[str] = None,
      accelerator_core_count: str = '0',
      labels: str = '',
      container_image_uri:
      str = 'gcr.io/deeplearning-platform-release/base-cpu:latest',
      params_yaml_file: Optional[str] = None,
      parameters: Optional[str] = None,
      block_pipeline: bool = True,
      fail_pipeline: bool = True
  ) -> NamedTuple('Outputs', [
      ('state', str),
      ('output_notebook_file', str),
      ('error', str),
  ]):
    """Creates the component.

    Args:
      project_id: str
      input_notebook_file: str
      output_notebook_folder: str
      execution_id: str
      location: str
      master_type: str
      scale_tier: str
      accelerator_type: str
      accelerator_core_count: str
      labels: str
      container_image_uri: str
      params_yaml_file: str
      parameters: str
      block_pipeline:bool whether to block the pipeline while waiting for the execution.
      fail_pipeline:bool whether to fail the pipeline if execution has error.
    """
    NotebooksExecutor.execute_notebook(**locals())

  @classmethod
  def execute_notebook(cls, *args, **kwargs) -> NamedTuple(
      'Outputs',
      [
          # ('GcpResource', str) # TODO(mayran): Add this everywhere: https://docs.google.com/document/d/1mzT3tY5xgbQMLkOr1r5IsKxy8XEnT7gsZQIfQbdxfRA/edit#heading=h.eu97jugvwzmf
          ('state', str),
          ('output_notebook_file', str),
          ('error', str),
      ],
  ):
    """Function that creates the component file `output_component_file`.

    Executes a notebooks and returns details about the execution. Function either
    returns a tuple or raises an error depending on the `fail_pipeline` parameter.

    Returns:
      state:str State of the execution. Generelly empty if error.
      output_notebook_file:str Executed notebook GCS path. Usually empty if error.
      error:str error message.

    Raises:
      RuntimeError with the error message.
    """
    cls._init_clients()

    # ------------------------------------------------
    # Sets input and outputs for the execution.
    # ------------------------------------------------

    project_id = kwargs.get('project_id', None)
    location = kwargs.get('location', None)
    execution_id = kwargs.get('execution_id', None)
    block_pipeline = kwargs.get('block_pipeline', None)
    fail_pipeline = kwargs.get('fail_pipeline', None)
    labels = [
      'vertex_pipelines=notebook_executor',
      kwargs.get('labels', ''),
    ]

    inputs_executor_api = SimpleNamespace(
      project_id=project_id,
      input_notebook_file=kwargs.get('input_notebook_file', None),
      output_notebook_folder=kwargs.get('output_notebook_folder', None),
      execution_id=kwargs.get('execution_id', None),
      location=location,
      master_type=kwargs.get('master_type', None),
      scale_tier=kwargs.get('scale_tier', None),
      accelerator_type=kwargs.get('accelerator_type', None),
      accelerator_core_count=kwargs.get('accelerator_core_count', None),
      labels=','.join([l for l in labels if l]),
      container_image_uri=kwargs.get('container_image_uri', None),
      params_yaml_file=kwargs.get('params_yaml_file', None),
      parameters=kwargs.get('parameters', None),
    )

    outputs = SimpleNamespace(
      state='',
      output_notebook_file = '',
      error = '',
    )

    # ------------------------------------------------
    # Runs execution using the Executor API.
    # ------------------------------------------------

    # Executes executions().create()
    ec_parent = f'projects/{project_id}/locations/{location}'
    ec_body = NotebooksExecutor._build_body_executor(inputs_executor_api)
    ec_request = cls._build_executions_create(ec_parent, execution_id, ec_body)
    ec_response, err = NotebooksExecutor._execute_executions_create(ec_request)
    if err:
      return NotebooksExecutor._build_notebook_execution_outputs(error=err)

    execution, err = cls._wait_for_executions_create(ec_response)
    if err:
      return NotebooksExecutor._build_notebook_execution_outputs(error=err)

    # Initial execution's parameters, after it was created by operation.
    if not block_pipeline:
      return NotebooksExecutor._build_notebook_execution_outputs(
          execution=execution)

    # Final execution's parameters, after it fully ran.
    execution_name = execution.get('name', '')
    execution, err = cls._wait_for_execution_run(execution_name)
    if err:
      return NotebooksExecutor._build_notebook_execution_outputs(error=err)

    # ------------------------------------------------
    # Returns outputs.
    # ------------------------------------------------
    return NotebooksExecutor._build_notebook_execution_outputs(
        execution=execution, error=err)

  # ----------------------------------
  # Helpers
  # ----------------------------------
  @staticmethod
  def _add_prefix(s, prefix='gs://'):
    """Adds the prefix gs:// to a GCS path when missing."""
    if not s.startswith(prefix):
      s = f'{prefix}{s}'
    return s

  @staticmethod
  def _handle_error(error_message, fail_pipeline=True):
    """Build the error logic.

    Manages how errors behave based on the fail_pipeline pipeline parameter.
    Either fails the pipeline by raising an error or sets the error for output.

    Args:
      error_message: str Error message.

    Raises:
      RuntimeError: with the error message if fail_pipeline is True.
    """
    if fail_pipeline:
      raise RuntimeError(error_message)
    return None, error_message

  @staticmethod
  def _build_body_executor(inputs):
    """Builds the body object for the Notebooks Executor API."""
    betpl = {}
    betpl['scaleTier'] = inputs.scale_tier
    betpl['masterType'] = inputs.master_type
    betpl['acceleratorConfig'] = {}
    betpl['inputNotebookFile'] = NotebooksExecutor._add_prefix(
        inputs.input_notebook_file)
    betpl['containerImageUri'] = inputs.container_image_uri
    betpl['outputNotebookFolder'] = NotebooksExecutor._add_prefix(
        inputs.output_notebook_folder)
    if inputs.labels:
      betpl['labels'] = dict(l.split('=') for l in inputs.labels.split(','))
    if inputs.accelerator_type:
      betpl['acceleratorConfig']['type'] = inputs.accelerator_type
      betpl['acceleratorConfig']['coreCount'] = inputs.accelerator_core_count
    if inputs.params_yaml_file:
      betpl['paramsYamlFile'] = inputs.params_yaml_file
    if inputs.parameters:
      betpl['parameters'] = inputs.parameters
    body = {}
    body['executionTemplate'] = betpl
    body['description'] = (
        f'Executor for notebook '
        f'{NotebooksExecutor._add_prefix(inputs.input_notebook_file)}')
    return body

  # -------------------------------------
  # API wrappers
  # -------------------------------------
  @classmethod
  def _init_clients(cls):
    """Initializes clients for Notebooks and ML APIs.

    TODO(mayran): Update when gapic available for Executor API.
    """
    http = httplib2.Http()
    http_user_agent = set_user_agent(
        http=http,
        user_agent='google-cloud-pipeline-components')
    client_notebooks = NotebooksExecutor._make_client_notebooks(http_user_agent)
    client_ml = NotebooksExecutor._make_client_ml(http_user_agent)
    cls.client_notebooks_locations = client_notebooks.projects().locations()
    cls.client_ml_jobs = client_ml.projects().jobs()

  @staticmethod
  def _make_client_notebooks(http):
    return discovery.build('notebooks', 'v1', http=http)

  @staticmethod
  def _make_client_ml(http):
    return discovery.build('ml', 'v1', http=http)

  @staticmethod
  def _execute_request(request):
    try:
      response = request.execute()
    except errors.HttpError as err:
      error = f'Error {err}'
      return NotebooksExecutor._handle_error(error)
    return response, None

  @classmethod
  def _build_executions_create(cls, parent, execution_id, body):
    return cls.client_notebooks_locations.executions().create(
        parent=parent,
        executionId=execution_id,
        body=body)

  @staticmethod
  def _execute_executions_create(request):
    return NotebooksExecutor._execute_request(request)

  @classmethod
  def _build_operations_get(cls, operation_name):
    return cls.client_notebooks_locations.operations().get(
        name=operation_name)

  @staticmethod
  def _wait_operations_get(request):
    while True:
      response, err = NotebooksExecutor._execute_request(request)
      if not response:
        return
      if response.get('done', False):
        break
      time.sleep(5)
    return response

  @classmethod
  def _build_executions_get(cls, execution_name):
    return cls.client_notebooks_locations.executions().get(
        name=execution_name)

  @staticmethod
  def _wait_executions_get(request):
    """Checks the state of the running execution until finished.

    Args:
      request: API request notebooks.project().locations().executions().get()

    Returns:
      An instance of Execution as seen in API.
    """
    while True:
      response = NotebooksExecutor._execute_request(request)
      if not response:
        return

      execution_state = response.get('state', None)
      if execution_state in ['SUCCEEDED', 'FAILED', 'CANCELLING', 'CANCELLED']:
        break
      time.sleep(30)
    return response

  @classmethod
  def _build_jobs_get(cls, job_name):
    return cls.client_ml_jobs.get(name=job_name)

  @staticmethod
  def _wait_jobs_get(request):
    while True:
      response = NotebooksExecutor._execute_request(request)
      if not response:
        return
      state = response.get('state', None)
      if state in ['SUCCEEDED', 'FAILED', 'CANCELLING', 'CANCELLED']:
        break
      time.sleep(30)
    return response

  # ----------------------------------
  # API calls
  # ----------------------------------
  @classmethod
  def _wait_for_executions_create(cls, operation):
    """Runs until an operation has 'done' == True.

    Waits for the operation that creates an execution finishes. It is different
    than waiting for the execution to finish (Done by _wait_for_execution). The
    pipeline is always blocked while the executions gets created. It should be
    quick enough.

    Args:
      operation: Operation object

    Returns:
      Tuple (execution:Execution, error:str) if fail_pipeline is False.

    Raises:
      RuntimeError with the error message if fail_pipeline is True.
    """
    operation_name = operation.get('name', None)
    if not operation_name:
      return NotebooksExecutor._handle_error('Operation is missing name.')

    request = cls._build_operations_get(operation_name)
    response = NotebooksExecutor._wait_operations_get(request)

    response_error = response.get('error', None)
    if response_error:
      error = f'Error {response_error["code"]}: {response_error["message"]}'
      return NotebooksExecutor._handle_error(error)
    execution = response.get('response', None)
    return execution, None

  @classmethod
  def _wait_for_execution_run(cls, execution_name):
    """Wait for a notebook execution to finish.

    Currently the notebooks executions().get() might returns a FAILED state but
    does not give errorMessage like the ml jobs().get() does. So we use the ml
    API to check the job as well. Error order is jobs > execution

    Args:
      execution_name: str The name of the execution expected by API of the form
        `projects/{project_id}/locations/{location}/executions/{execution_id}`.

    Returns:
      Tuple (execution:Execution, error:str) if fail_pipeline is False

    Raises:
      RuntimeError with the error message if fail_pipeline is True.
    """
    # First tries to get the Execution state, returns if API call errors.
    eg_request = cls._build_executions_get(execution_name)
    eg_response = NotebooksExecutor._wait_executions_get(eg_request)
    eg_state = eg_response.get('state', None)

    # For some reason, execution.get and job.get might not return the same state
    # so we check if we can get the error message using the ml job API.
    if eg_state in ['FAILED', 'CANCELLING', 'CANCELLED']:
      project_id = execution_name.split('/')[1]
      execution_id = execution_name.split('/')[-1]
      job_name = f'projects/{project_id}/jobs/{execution_id}'

      jg_request = cls._build_jobs_get(job_name)
      jg_response = NotebooksExecutor._wait_jobs_get(jg_request)

      error_message = jg_response.get('errorMessage', None)
      if error_message:
        return NotebooksExecutor._handle_error(error_message)

      # The job might be successful but we need to address that the execution
      # had a problem. The previous loop was in hope to find the error message,
      # we didn't have any so we return the execution state as the message.
      return NotebooksExecutor._handle_error(
          f'Execution finished with state: {eg_state}')

    return eg_response, None

  @staticmethod
  def _build_notebook_execution_outputs(execution={}, error=''):
    """Creates outputs values based on error and an Execution object."""
    return (
      execution.get('state', ''),
      execution.get('outputNotebookFile', ''),
      error,
    )

if __name__ == '__main__':
  notebooks_executor_op = create_component_from_func(
      NotebooksExecutor.execute_notebook,
      base_image='python:3.8',
      packages_to_install=['google-api-python-client'],
      output_component_file='component.yaml',
      annotations={
        "author": "Matthieu Mayran",
        "canonical_location": "https://raw.githubusercontent.com/Ark-kun/pipeline_components/master/components/google-cloud/experimenal/notebooks/component.yaml",
      },
      # TODO(mayran): Wait for it to be available in kfp api
      # output_image='gcr.io/ml-pipeline/notebook-executor:v0.1',
  )
