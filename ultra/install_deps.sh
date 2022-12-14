#!/usr/bin/env bash

function check_python_version_gte_3_7 {
    echo "Checking for >=python3.7"
    # running through current minor verions
    hash python3.7 2>/dev/null \
    || hash python3.8 2>/dev/null \
    || hash python3.9 2>/dev/null;
}

# function check_python_version_gte_3_8 {
#     echo "Checking for >=python3.8"
#     # running through current minor verions
#     hash python3.8 2>/dev/null \
#     || hash python3.9 2>/dev/null;
# }

function do_install_for_linux {
    echo "Installing sumo (used for traffic simulation and road network)"
    sudo add-apt-repository ppa:sumo/stable
    sudo apt-get update

    sudo apt-get install -y \
         libspatialindex-dev \
         sumo sumo-tools sumo-doc \
         build-essential cmake

    #only a problem for linux
    if ! check_python_version_gte_3_7; then

         echo "A >=3.7 python version not found"
         read -p "Install python3.7? [Yn]" should_add_python_3_7
         if [[ $should_add_python_3_7 =~ ^[yY\w]*$ ]]; then
              echo ""
              printf "This will run the following commands:\n$ sudo apt-get update\n$ sudo apt-get install software-properties-common\n$ sudo add-apt-repository ppa:deadsnakes/ppa\n$ sudo apt-get install python3.7 python3.7-dev python3.7-tk python3.7-venv"
              echo ""
              read -p "WARNING. Is this OK? If you are unsure choose no. [Yn]" should_add_python_3_7
              # second check to make sure they really want to
              if [[ $should_add_python_3_7 =~ ^[yY\w]*$ ]]; then
                    sudo apt-get install software-properties-common
                    sudo add-apt-repository ppa:deadsnakes/ppa
                    sudo apt-get install python3.7 python3.7-dev python3.7-tk python3.7-venv
              fi
         fi
    fi

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
    echo "You'll need to set the SUMO_HOME variable. Logging out and back in will"
    echo "get you set up. Alternatively, in your current session, you can run:"
    echo ""
    echo "  source /etc/profile.d/sumo.sh"
    echo ""
}

function do_install_for_macos {
    echo "Installing sumo (used for traffic simulation and road network)"
    brew tap dlr-ts/sumo
    brew install sumo spatialindex # for sumo
    brew install geos # for shapely

    # start X11 manually the first time, logging in/out will also do the trick
    open -g -a XQuartz.app

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
    read -p "Add SUMO_HOME to ~/.bash_profile? [Yn]" should_add_SUMO_HOME
    echo "should_add_SUMO_HOME $should_add_SUMO_HOME"
    if [[ $should_add_SUMO_HOME =~ ^[yY\w]*$ ]]; then
        echo 'export SUMO_HOME="/usr/local/opt/sumo/share/sumo"' >> ~/.bash_profile
        echo "We've updated your ~/.bash_profile. Be sure to run:"
        echo ""
        echo "  source ~/.bash_profile"
        echo ""
        echo "in order to set the SUMO_HOME variable in your current session"
    else
        echo "Not updating ~/.bash_profile"
        echo "Make sure SUMO_HOME is set before proceeding"
    fi
}

function do_install_for_WSL {
    # We currently only support Ubuntu distribution (v18.04) 
    # for WSL (Windows subsystem for Linux). 
    echo "Installing in WSL"
    
    local UBUNTU_VERSION=$(lsb_release -rs)
    if [[ $UBUNTU_VERSION == "18.04" ]]; then
    
        echo "Installing sumo (used for traffic simulation and road network)"
        sudo add-apt-repository ppa:sumo/stable
        sudo apt-get update

        sudo apt-get install -y \
            libspatialindex-dev \
            sumo sumo-tools sumo-doc \
            build-essential cmake

        #only a problem for linux
        if ! check_python_version_gte_3_7; then

            echo "A >=3.7 python version not found"
            read -p "Install python3.7? [Yn]" should_add_python_3_7
            if [[ $should_add_python_3_7 =~ ^[yY\w]*$ ]]; then
                echo ""
                printf "This will run the following commands:\n$ sudo apt-get update\n$ sudo apt-get install software-properties-common\n$ sudo add-apt-repository ppa:deadsnakes/ppa\n$ sudo apt-get install python3.7 python3.7-dev python3.7-tk python3.7-venv"
                echo ""
                read -p "WARNING. Is this OK? If you are unsure choose no. [Yn]" should_add_python_3_7
                # second check to make sure they really want to
                if [[ $should_add_python_3_7 =~ ^[yY\w]*$ ]]; then
                        sudo apt-get install software-properties-common
                        sudo add-apt-repository ppa:deadsnakes/ppa
                        sudo apt-get install python3.7 python3.7-dev python3.7-tk python3.7-venv
                fi
            fi
        fi
    else
        echo "Unsupported UBUNTU RELEASE: $UBUNTU_VERSION"
        exit 1
    fi

    echo ""
    echo "-- dependencies have been installed --"
    echo ""
    echo "You'll need to set the SUMO_HOME variable. Logging out and back in will"
    echo "get you set up. Alternatively, in your current session, you can run:"
    echo ""
    echo "  source /etc/profile.d/sumo.sh"
    echo ""
}

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    echo "Detected Linux"
    # WSL1 and WSL2 both have different kernal version string
    # representations: For example
    #   WSL1: 4.X.X-12345-Microsoft
    #   WSL2: 5.XX.XX.X-microsoft-standard-WSL2
    # To ensure that both versions of WSL are supported, the
    # kernal release string needs to be compared with keywords
    # from WSL1 (i.e Microsoft) and WSL2 (i.e WSL2) kernal 
    # version string formats. 
    if [[ $(uname -r) =~ (Microsoft|WSL2) ]]; then
        do_install_for_WSL
    else
        do_install_for_linux
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    do_install_for_macos
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi
