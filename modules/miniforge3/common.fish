set-function __conda_add_prompt {
    if set --query CONDA_PROMPT_MODIFIER
        set_color --bold green
        echo -n $CONDA_PROMPT_MODIFIER
        set_color normal
    end
}

# if functions --query fish_prompt
#     if not functions --query __fish_prompt_orig
#         functions --copy fish_prompt __fish_prompt_orig
#     end
#     functions --erase fish_prompt
# else
#     function __fish_prompt_orig
#     end
# end

# set-function return_last_status {
#     return $argv
# }

# set-function fish_prompt {
#     set --local last_status $status
#     if set --query CONDA_LEFT_PROMPT
#         __conda_add_prompt
#     end
#     return_last_status $last_status
#     __fish_prompt_orig
# }

# if functions --query fish_right_prompt
#     if not functions --query __fish_right_prompt_orig
#         functions --copy fish_right_prompt __fish_right_prompt_orig
#     end
#     functions --erase fish_right_prompt
# else
#     function __fish_right_prompt_orig
#     end
# end

# set-function fish_right_prompt {
#     if not set --query CONDA_LEFT_PROMPT
#         __conda_add_prompt
#     end
#     __fish_right_prompt_orig
# }

set-function __rehash_modules {
	set --local list (string split : $LOADEDMODULES)
    command cat $PYENV_ROOT/version | read THIS_VER
	set --local THIS (string replace - / $THIS_VER)
    set --local normal ()
    set --local inverse ()
    for target in $list
        if [ $target != anyenv -a $target != $THIS ]
            set --prepend inverse $target
            set --append normal $target
		end
    end
    for target in $inverse
		module unload $target
    end
    for target in $normal
		module load $target
    end
}

set-function __conda_activate {
    eval ($CONDA_EXE shell.fish $argv)
    __rehash_modules
}

# set-function conda --inherit-variable CONDA_EXE{
set-function conda {
        if [ (count $argv) -lt 1 ]
        $CONDA_EXE
    else
        set --local cmd $argv[1]
        set --erase argv[1]
        switch $cmd
            case activate deactivate
                # eval ($CONDA_EXE shell.fish $cmd $argv)
                __conda_activate $cmd $argv
            case install update upgrade remove uninstall
                $CONDA_EXE $cmd $argv
                and eval ($CONDA_EXE shell.fish reactivate)
            case '*'
                $CONDA_EXE $cmd $argv
        end
    end
}

# Autocompletions below
# Faster but less tested (?)
set-function __fish_conda_commands {
    string replace --regex '.*_([a-z]+)\.py$' '$1' $CONDA_ROOT/lib/python*/site-packages/conda/cli/main_*.py
    for f in $CONDA_ROOT/bin/conda-*
        if test -x "$f" -a ! -d "$f"
            string replace --regex '^.*/conda-' '' "$f"
        end
    end
    echo activate
    echo deactivate
}

set-function __fish_conda_env_commands {
    string replace --regex '.*_([a-z]+)\.py$' '$1' $CONDA_ROOT/lib/python*/site-packages/conda_env/cli/main_*.py
}

set-function __fish_conda_envs {
    conda config --json --show envs_dirs | python -c "import json, os, sys; from os.path import isdir, join; print('\n'.join(d for ed in json.load(sys.stdin)['envs_dirs'] if isdir(ed) for d in os.listdir(ed) if isdir(join(ed, d))))"
}

set-function __fish_conda_packages {
    conda list | awk 'NR > 3 {print $1}'
}

set-function __fish_conda_needs_command {
    set cmd (commandline -opc)
    if [ (count $cmd) -eq 1 -a $cmd[1] = conda ]
        return 0
    end
    return 1
}

set-function __fish_conda_using_command {
    set cmd (commandline -opc)
    if [ (count $cmd) -gt 1 ]
        if [ $argv[1] = $cmd[2] ]
            return 0
        end
    end
    return 1
}

# # Conda commands
# complete --no-files --command conda --condition __fish_conda_needs_command --arguments '(__fish_conda_commands)'
# complete --no-files --command conda --condition '__fish_conda_using_command env' --arguments '(__fish_conda_env_commands)'

# # Commands that need environment as parameter
# complete --no-files --command conda --condition '__fish_conda_using_command activate' --arguments '(__fish_conda_envs)'

# # Commands that need package as parameter
# complete --no-files --command conda --condition '__fish_conda_using_command remove' --arguments '(__fish_conda_packages)'
# complete --no-files --command conda --condition '__fish_conda_using_command uninstall' --arguments '(__fish_conda_packages)'
# complete --no-files --command conda --condition '__fish_conda_using_command upgrade' --arguments '(__fish_conda_packages)'
# complete --no-files --command conda --condition '__fish_conda_using_command update' --arguments '(__fish_conda_packages)'
