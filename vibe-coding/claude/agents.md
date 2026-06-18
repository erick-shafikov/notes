настройки

```yaml
# название - обязательное
name: code-reviewer # должно совпадать
description: Reviews code for quality and best practices
tools: Read, Glob, Grep
model: sonnet # sonnet, opus, haiku, fable
tools: AskUserQuestion # EnterPlanMode ExitPlanMode ScheduleWakeup WaitForMcpServers Agent(worker, researcher) - если работает claude --agent
disallowedTools: mcp__github # отключить
permissionMode: default, # acceptEdits, auto, dontAsk, bypassPermissions или plan
maxTurns: 1
skills: SkillName # skill который будет загружен в контекст
skills:
  - SkillName1
  - SkillName2
mcpServers: MCPServerName
mcpServers:
- playwright:
      type: stdio
      command: npx
      args: ["-y", "@playwright/mcp@latest"]
hooks: PreToolUse # PostToolUse Stop
hooks:
  PreToolUse:
      - matcher: "Bash"
        hooks:
          - type: command
            command: "./scripts/validate-readonly-query.sh"
background: false # запускать как фоновую задачу
effort: low # medium, high, xhigh, max (зависит от модели)
isolation: worktree
color: red # blue, green, yellow, purple, orange, pink или cyan
initialPrompt: some prompt # добавляется при прямом вызове
memory: user # - во всех проектах project - для проекта local - не будет попадать в git
```

You are a code reviewer. When invoked, analyze the code and provide
specific, actionable feedback on quality, security, and best practices.
