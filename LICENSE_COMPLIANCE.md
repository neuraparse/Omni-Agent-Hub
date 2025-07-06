# License Compliance Guide

## 📋 **Overview**

This document provides guidance on license compliance when using, modifying, or distributing Omni-Agent Hub.

## ⚖️ **MIT License Requirements**

### ✅ **What You Can Do**
- **Commercial Use**: Use in commercial projects
- **Modification**: Modify the source code
- **Distribution**: Distribute original or modified versions
- **Private Use**: Use privately without disclosure
- **Patent Use**: Use any patents that may be applicable

### 📋 **What You Must Do**
- **Include License**: Include the MIT license text
- **Include Copyright**: Include the copyright notice
- **Document Changes**: Document significant modifications (recommended)

### ❌ **What You Cannot Do**
- **Hold Liable**: Hold authors liable for damages
- **Use Trademarks**: Use project trademarks without permission

## 📄 **Required Notices**

### **Minimum Attribution**
When distributing Omni-Agent Hub, include:

```
Copyright (c) 2025 Neuraparse
Licensed under MIT License
```

### **Full Attribution**
For complete compliance, include:

1. **LICENSE file**: The complete MIT license text
2. **COPYRIGHT notice**: Copyright (c) 2025 Neuraparse
3. **NOTICE file**: Third-party attributions (if applicable)

## 🔧 **Integration Scenarios**

### **Scenario 1: Using as a Library**
```python
# Include in your project documentation:
# This project uses Omni-Agent Hub
# Copyright (c) 2025 Neuraparse
# Licensed under MIT License
```

### **Scenario 2: Modifying the Code**
- Include original copyright notice
- Add your own copyright for modifications
- Document changes made
- Include MIT license text

### **Scenario 3: Commercial Distribution**
- Include LICENSE file in distribution
- Include COPYRIGHT notice
- Include NOTICE file for third-party components
- Consider trademark usage guidelines

## 🏢 **Enterprise Compliance**

### **Legal Review Checklist**
- [ ] MIT license terms reviewed
- [ ] Third-party dependencies audited
- [ ] Copyright notices included
- [ ] Attribution requirements met
- [ ] Export control compliance verified

### **Recommended Practices**
- **License Scanning**: Use automated license scanning tools
- **Legal Review**: Have legal team review usage
- **Documentation**: Document all dependencies and licenses
- **Regular Audits**: Perform regular compliance audits

## 🔍 **Third-Party Dependencies**

### **License Compatibility Matrix**

| Component | License | Compatible | Notes |
|-----------|---------|------------|-------|
| FastAPI | MIT | ✅ Yes | Fully compatible |
| PostgreSQL | PostgreSQL | ✅ Yes | Compatible with MIT |
| Redis | BSD 3-Clause | ✅ Yes | Compatible with MIT |
| Kafka | Apache 2.0 | ✅ Yes | Compatible with MIT |
| Milvus | Apache 2.0 | ✅ Yes | Compatible with MIT |
| MinIO | AGPL v3 | ⚠️ Caution | Network use only |
| OpenAI SDK | MIT | ✅ Yes | Fully compatible |

### **Special Considerations**

#### **MinIO (AGPL v3)**
- **Network Use**: Safe for network services
- **Distribution**: Requires source code disclosure if distributed
- **Recommendation**: Use as separate service

#### **OpenAI API**
- **Terms of Service**: Subject to OpenAI's terms
- **Data Processing**: Review OpenAI's data usage policies
- **Commercial Use**: Allowed under OpenAI's terms

## 📊 **Compliance Tools**

### **Recommended Tools**
- **FOSSA**: License compliance scanning
- **Black Duck**: Open source security and license compliance
- **WhiteSource**: Open source security and license management
- **Snyk**: Security and license scanning

### **Manual Verification**
```bash
# Check all dependencies
pip-licenses --format=table

# Generate license report
pip-licenses --format=json --output-file=licenses.json
```

## 🚨 **Common Compliance Issues**

### **Missing Attribution**
- **Problem**: Not including copyright notices
- **Solution**: Include all required notices
- **Prevention**: Automated compliance checking

### **License Conflicts**
- **Problem**: Incompatible license combinations
- **Solution**: Review and replace incompatible components
- **Prevention**: License compatibility matrix

### **Trademark Violations**
- **Problem**: Using project trademarks incorrectly
- **Solution**: Follow trademark usage guidelines
- **Prevention**: Legal review of marketing materials

## 📞 **Getting Help**

### **Legal Questions**
- **GitHub Issues**: Create issue with "legal" label
- **Legal Review**: Consult with legal counsel
- **Community**: Ask in GitHub Discussions

### **License Clarifications**
- **MIT License**: https://opensource.org/licenses/MIT
- **OSI Approved**: https://opensource.org/licenses/
- **License Compatibility**: https://www.gnu.org/licenses/license-list.html

## 🔄 **Updates and Changes**

### **Monitoring Changes**
- **Watch Repository**: Get notified of license changes
- **Review Releases**: Check changelog for legal updates
- **Dependency Updates**: Monitor dependency license changes

### **Version Control**
- **License History**: Track license changes in git
- **Compliance Records**: Maintain compliance documentation
- **Regular Reviews**: Schedule periodic compliance reviews

## 📋 **Compliance Checklist**

### **Before Using Omni-Agent Hub**
- [ ] Review MIT license terms
- [ ] Understand attribution requirements
- [ ] Check third-party dependency licenses
- [ ] Verify export control compliance
- [ ] Consult legal counsel if needed

### **During Development**
- [ ] Include copyright notices in code
- [ ] Document modifications made
- [ ] Track dependency changes
- [ ] Maintain license documentation

### **Before Distribution**
- [ ] Include LICENSE file
- [ ] Include COPYRIGHT notice
- [ ] Include NOTICE file
- [ ] Verify all attributions
- [ ] Test compliance scanning

---

**Disclaimer**: This guide is for informational purposes only and does not constitute legal advice. Consult with qualified legal counsel for specific compliance questions.

**Last Updated**: July 6, 2025
