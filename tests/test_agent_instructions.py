"""Participant instructions have one source and protected Claude edit paths."""
import json
from pathlib import Path
import subprocess
import unittest

REPO=Path(__file__).resolve().parents[1]


class InstructionTests(unittest.TestCase):
    def test_generated_generic_is_exact_and_generation_check_passes(self):
        self.assertEqual((REPO/'AGENTS.md').read_bytes(),(REPO/'docs/agents/generic.md').read_bytes())
        import sys
        subprocess.run([sys.executable,str(REPO/'scripts/sync_agent_instructions.py'),'--check'],check=True)

    def test_claude_import_and_protected_paths(self):
        self.assertIn('@AGENTS.md',(REPO/'CLAUDE.md').read_text())
        deny=json.loads((REPO/'.claude/settings.json').read_text())['permissions']['deny']
        for prefix in ('splits','protocols','grmhd_validation','references'):
            for tool in ('Edit','Write'):self.assertIn(tool+'(/'+prefix+'/**)',deny)
        instructions=(REPO/'AGENTS.md').read_text()
        for text in ('--tier quick','check-adapter','gval verify','never cancel','NOT_EVALUATED','prediction_roles'):
            self.assertIn(text,instructions)
        skill=(REPO/'.claude/skills/validate-model/SKILL.md').read_text()
        for command in ('doctor','conformance','new-adapter','check-adapter','verify'):self.assertIn(command,skill)
