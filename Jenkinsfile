pipeline {
  agent any
  parameters {
    string(name: 'SERVICE_REF', defaultValue: 'develop', description: 'Branch/tag/SHA')
  }
  stages {
    stage('Checkout') {
      steps { git branch: params.SERVICE_REF, url: 'https://github.com/PlataformaIntegradaInvestigadores/predictive_model_backend.git' }
    }
    stage('Quality Gate') {
      steps {
        sh 'ruff check . && black --check .'
        sh 'pytest --cov --cov-fail-under=0'
      }
    }
    stage('Build')       { steps { sh 'docker compose build' } }
    stage('Deploy')      { steps { sh 'docker compose up -d' } }
    stage('Healthcheck') { steps { sh 'curl -f http://localhost:8003/health || exit 1' } }
    stage('Manifest')    { steps { sh 'echo "MANIFEST update: predictive SHA=$GIT_COMMIT"' } }
  }
  post { failure { echo 'Deploy failed. Revisar logs.' } }
}
