export ZSH="/root/.oh-my-zsh"

HYPHEN_INSENSITIVE="true"
DISABLE_AUTO_UPDATE="true"
DISABLE_UPDATE_PROMPT="true"
ZSH_THEME="Soliah"


plugins=()

alias pycpu="JAX_PLATFORM_NAME=cpu python"

autoload -Uz compinit && compinit
source $ZSH/oh-my-zsh.sh
