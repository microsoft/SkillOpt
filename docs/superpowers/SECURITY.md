# Security Considerations for Superpowers Adapter

## Execution Model

The Superpowers adapter runs Claude Code with candidate skills that control agent behavior. A malicious skill can:

- Execute arbitrary shell commands
- Read/write files in the project directory
- Access environment variables (including API keys)
- Make network requests

## Current Mitigations

1. **Scrubbed environment**: Only essential vars passed (HOME, PATH, TERM, LANG, ANTHROPIC_API_KEY)
2. **Isolated HOME**: Each scenario gets its own HOME directory
3. **No `--dangerously-skip-permissions` by default**: Permission prompts required unless explicitly bypassed
4. **Project directory isolation**: Each scenario gets its own project directory

## Known Limitations

- **API key exposure**: ANTHROPIC_API_KEY is passed to the subprocess
- **No OS-level isolation**: Without Docker/bubblewrap, candidate code runs with user privileges
- **SKILLOPT_UNSAFE bypass**: When enabled, full filesystem access

## Recommendations

### For Local Testing (trusted candidates)
```bash
SKILLOPT_UNSAFE=1 python -m skillopt_sleep.adapters.superpowers --candidate my_skill.md
```

### For Untrusted Candidates (future work)

Docker isolation (not yet implemented):
```bash
# Build sandbox image
docker build -t skillopt-sandbox .

# Run with isolation
python -m skillopt_sleep.adapters.superpowers --candidate untrusted.md --sandbox docker
```

## Follow-up Work

- [ ] Docker sandbox implementation
- [ ] Bubblewrap (bwrap) support for Linux
- [ ] Network isolation option
- [ ] API key injection via Docker secrets
