SESSIONNAME="bsky-umap"
tmux has-session -t $SESSIONNAME &> /dev/null

if [ $? != 0 ]
 then
    tmux new-session -s $SESSIONNAME -n $SESSIONNAME -d
fi

tmux attach -t $SESSIONNAME
