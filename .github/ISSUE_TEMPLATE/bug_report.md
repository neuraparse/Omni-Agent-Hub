---
name: Bug Report
about: Create a report to help us improve Omni-Agent Hub
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce

Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## ✅ Expected Behavior

A clear and concise description of what you expected to happen.

## ❌ Actual Behavior

A clear and concise description of what actually happened.

## 📸 Screenshots

If applicable, add screenshots to help explain your problem.

## 🖥️ Environment

**System Information:**
- OS: [e.g. macOS 14.0, Ubuntu 22.04, Windows 11]
- Python Version: [e.g. 3.11.5]
- Docker Version: [e.g. 24.0.6]
- Docker Compose Version: [e.g. 2.21.0]

**Omni-Agent Hub:**
- Version: [e.g. v0.0.1]
- Installation Method: [e.g. pip install, docker-compose]
- Configuration: [e.g. development, production]

**Services Status:**
- [ ] PostgreSQL
- [ ] Redis  
- [ ] Milvus
- [ ] Kafka
- [ ] MinIO
- [ ] OpenAI API

## 📋 Configuration

**Environment Variables (remove sensitive data):**
```env
DEBUG=true
LOG_LEVEL=DEBUG
OPENAI_MODEL=gpt-4o
# Add relevant config without API keys
```

## 📊 Logs

**Error Logs:**
```
Paste relevant error logs here
```

**System Health Check:**
```bash
# Output of: curl http://localhost:8000/api/v1/system/status
```

## 🔍 Additional Context

Add any other context about the problem here.

## 🚀 Possible Solution

If you have ideas on how to fix this, please describe them here.

## ✅ Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided all the requested information
- [ ] I have tested this with the latest version
- [ ] I have checked the system health status
- [ ] I have included relevant logs and error messages
