#!/bin/bash

# Define variables
LOCAL_DIR="/Users/mahdi/Documents/Research/DT_QGT"  # Change this to your local directory
REMOTE_USER="mahdis"       # Change this to your server username
REMOTE_HOST="hpcfund.amd.com" # Change this to your server address (without username)
REMOTE_DIR="/work1/javidi/mahdis/DT_QGT"  # Use \$WORK so it expands on the remote machine

# Copy files from local to remote, overwriting if they exist
scp -r "$LOCAL_DIR"/* "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo "Files successfully copied to the server!"