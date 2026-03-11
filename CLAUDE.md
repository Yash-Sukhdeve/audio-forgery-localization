# CLAUDE.md

Project-specific instructions for Claude Code.

<!-- UWS-BEGIN -->
## UWS Workflow System

This project uses UWS (Universal Workflow System) for context persistence across sessions.

### Commands
- `/uws` - Show all available UWS commands
- `/uws-status` - Show current workflow state
- `/uws-checkpoint "msg"` - Create checkpoint
- `/uws-recover` - Full context recovery after break
- `/uws-handoff` - Prepare handoff before ending session
- `/uws-sdlc <action>` - Manage SDLC phases (status/start/next/goto/fail/reset)
- `/uws-research <action>` - Manage research phases (status/start/next/goto/reject/reset)

### Workflow Files
- `.workflow/state.yaml` - Current phase and checkpoint
- `.workflow/handoff.md` - Human-readable context (READ THIS ON SESSION START)
- `.workflow/checkpoints.log` - Checkpoint history

### Session Workflow
1. **Start**: Context is automatically loaded via SessionStart hook
2. **During**: Create checkpoints at milestones with `/uws-checkpoint`
3. **End**: Run `/uws-handoff` to update context for next session

### Auto-Checkpoint
UWS automatically creates checkpoints before context compaction to prevent state loss.
<!-- UWS-END -->

