pipeline {
    agent any
    parameters {
        choice(name: 'STAGE_TO_RUN', choices: ['Create Virtual Environment', 'Run MLflow', 'Run Embedding Script', 'Run Main Script', 'Cleanup'], description: 'Choose which stage to execute')
    }
    environment {
        PATH = "/Users/nileshkishore/anaconda3/bin:$PATH"
    }
    stages {
        stage('Create Virtual Environment') {
            when {
                expression { return params.STAGE_TO_RUN == 'Create Virtual Environment' }
            }
            steps {
                script {
                    // Ensure we're operating in the current workspace
                    def workspaceDir = pwd()
                    echo "Current Workspace: ${workspaceDir}"

                    // Check if the virtual environment exists in the workspace
                    def venvExists = fileExists("${workspaceDir}/venv")
                    if (!venvExists) {
                        // If venv doesn't exist, create it
                        sh """
                            python3 -m venv ${workspaceDir}/venv
                            source ${workspaceDir}/venv/bin/activate
                            which python3
                            python3 --version
                        """
                    } else {
                        echo "Virtual environment already exists."
                    }
                }
            }
        }

        stage('Install Dependencies') {
            when {
                expression { return params.STAGE_TO_RUN == 'Create Virtual Environment' || params.STAGE_TO_RUN == 'Run Embedding Script' || params.STAGE_TO_RUN == 'Run Main Script' }
            }
            steps {
                script {
                    // Ensure we're operating in the current workspace
                    def workspaceDir = pwd()
                    echo "Current Workspace: ${workspaceDir}"

                    // Install dependencies in the current workspace's venv
                    sh """
                        source ${workspaceDir}/venv/bin/activate
                        pip install --upgrade pip
                        pip install -r ${workspaceDir}/requirements.txt
                    """
                }
            }
        }

        stage('Run MLflow') {
            when {
                expression { return params.STAGE_TO_RUN == 'Run MLflow' }
            }
            steps {
                script {

                    // Run MLflow UI on a specific port (5000) using the workspace directory
                    sh """
                        source ${workspaceDir}/venv/bin/activate
                        nohup mlflow ui
                        echo "MLflow server is running on port 5000 in the background."
                    """
                }
            }
        }


        stage('Run Embedding Script') {
            when {
                expression { return params.STAGE_TO_RUN == 'Run Embedding Script' }
            }
            steps {
                script {
                    def workspaceDir = pwd()
                    echo "Current Workspace: ${workspaceDir}"

                    // Run the embedding script in the current workspace's venv
                    sh """
                        source ${workspaceDir}/venv/bin/activate
                        python3 ${workspaceDir}/embedding.py
                    """
                }
            }
        }

        stage('Run Main Script') {
            when {
                expression { return params.STAGE_TO_RUN == 'Run Main Script' }
            }
            steps {
                script {
                    // Get the current workspace directory dynamically
                    def workspaceDir = pwd()
                    echo "Using current workspace directory: ${workspaceDir}"

                    // Run MLflow UI in the background on port 5000
                    sh """
                        source ${workspaceDir}/venv/bin/activate
                        nohup mlflow ui --port 5003 > ${workspaceDir}/mlflow.log 2>&1 &
                        echo "MLflow server is running on port 5003 in the background."
                    """

                    // Now, run the main.py script in the current workspace's venv
                    //python3 ${workspaceDir}/main.py
                    sh """
                        source ${workspaceDir}/venv/bin/activate
                        streamlit run ${workspaceDir}/app.py 
                    """
                }
            }
        }
        stage('Cleanup') {
            when {
                expression { return params.STAGE_TO_RUN == 'Cleanup' }
            }
            steps {
                script {
                    def workspaceDir = pwd()
                    echo "Current Workspace: ${workspaceDir}"

                    // Clean up: Kill the MLflow server and remove virtual environment
                    echo "Cleaning up..."
                    
                    // Kill MLflow server process
                    sh """
                        pkill -f 'mlflow ui' || echo 'No MLflow process found to kill'
                    """
                    
                    // Remove virtual environment in the current workspace
                    echo "Removing virtual environment..."
                    sh "rm -rf ${workspaceDir}/venv"
                    echo "Cleanup completed!"
                }
            }
        }
    }
    post {
        always {
            echo "Pipeline execution completed"
        }
    }
}
