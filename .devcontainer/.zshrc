export ZSH="/root/.oh-my-zsh"

HYPHEN_INSENSITIVE="true"
DISABLE_AUTO_UPDATE="true"
DISABLE_UPDATE_PROMPT="true"
ZSH_THEME="Soliah"

plugins=(
    git
    history-substring-search
    sudo
    zsh-autosuggestions
    zsh-syntax-highlighting
)

alias pycpu="JAX_PLATFORM_NAME=cpu python"
alias pytest="JAX_PLATFORM_NAME=cpu pytest"

source $ZSH/oh-my-zsh.sh