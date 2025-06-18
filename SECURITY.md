# Security Guidelines

## API Key Management

### ⚠️ CRITICAL: Never commit API keys to version control

This project requires OpenAI API keys for functionality. **NEVER** commit API keys or other secrets to the repository.

### Proper API Key Setup

1. **Use environment variables:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

2. **Use .env files (not committed):**
```bash
# Create .env file (ignored by git)
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

3. **Use the provided .env.example template:**
```bash
cp .env.example .env
# Edit .env with your actual API key
```

### What's Protected

The `.gitignore` file prevents committing:
- `.env` and `.env.*` files
- Files with `*.key`, `*.secret` extensions
- Files starting with `sk-` (OpenAI API key format)
- Common secret file names (`credentials.json`, `secrets.json`, etc.)

### If You Accidentally Commit an API Key

1. **Immediately revoke the key** at https://platform.openai.com/api-keys
2. **Generate a new API key**
3. **Remove the key from git history:**
```bash
# Remove from most recent commit
git reset --soft HEAD~1
git reset HEAD .
git add . --ignore-removal
git commit -m "Remove leaked API key"

# For keys in git history, use:
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch filename_with_key' \
--prune-empty --tag-name-filter cat -- --all
```

### Code Examples

**✅ Correct - Using environment variables:**
```python
import os
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
```

**❌ Wrong - Hardcoded API key:**
```python
api_key = "sk-proj-abc123..."  # NEVER DO THIS
```

**✅ Correct - Check for key presence:**
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
```

## Security Features in the Framework

### Input Validation
The framework includes security validation for:
- **Prompt injection detection**: Blocks attempts to override instructions
- **PII detection**: Identifies Social Security Numbers, credit cards, etc.
- **Credential exposure**: Detects accidentally included passwords/API keys
- **XSS pattern detection**: Blocks HTML/JavaScript injection attempts

### Example Security Patterns
```python
security_patterns = [
    r"(?i)ignore.*previous.*instructions",    # Prompt injection
    r"(?i)system.*prompt",                    # System prompt extraction
    r"(?i)override.*safety",                  # Safety override attempts
    r"(?i)ssn.*\d{3}-\d{2}-\d{4}",          # Social Security Numbers
    r"(?i)api_key.*sk-",                     # API key exposure
    r"(?i)password.*:"                       # Password exposure
]
```

### Usage Tracking Security
- **Cost limits**: Prevent unexpectedly high API usage
- **Rate limiting**: Prevent abuse through rapid requests
- **Usage monitoring**: Track and alert on unusual patterns

## Best Practices

### Development
1. **Always use environment variables** for API keys
2. **Use .env files** for local development (ensure they're gitignored)
3. **Validate inputs** before sending to LLM APIs
4. **Implement rate limiting** to prevent abuse
5. **Monitor usage** for unexpected patterns

### Production
1. **Use secure secret management** (AWS Secrets Manager, HashiCorp Vault, etc.)
2. **Implement logging** that excludes sensitive data
3. **Use HTTPS** for all API communications
4. **Regular key rotation** for enhanced security
5. **Monitor for leaked keys** in public repositories

### Incident Response
If you discover a security issue:
1. **Don't commit the fix** with details in the commit message
2. **Report privately** to maintainers first
3. **Allow time** for coordinated disclosure
4. **Document lessons learned** after resolution

## Compliance

This framework helps with:
- **GDPR compliance**: PII detection and redaction
- **SOC 2**: Audit logging and access controls
- **HIPAA**: Data protection for healthcare applications
- **Financial regulations**: Audit trails and data protection

## Contact

For security issues, please contact the maintainers privately rather than opening public issues.

---

**Remember: Security is everyone's responsibility. When in doubt, err on the side of caution.**