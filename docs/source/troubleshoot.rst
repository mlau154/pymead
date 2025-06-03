Troubleshooting
###############

This section addresses common issues which may arise when using *pymead*.

* Why can't I move the dialog windows?

    * When using the Linux version of *pymead*, you may find that the dialog pop-ups only
      move with the main window. This is the default behavior in some flavors of Linux. To change this behavior
      in Ubuntu 20.04 or newer, run the command shown below in a terminal. In other versions of Linux,
      the key setting to disable is "Attach Modal Dialogs."

      .. code-block:: console

         gsettings set org.gnome.mutter attach-modal-dialogs false

* I get a message like ``This plugin supports grabbing the mouse only for popup windows`` or
  ``xdg_wm_base#3: error 0: Surface already has a role.``

    * This issue has been found when using the Wayland version of Qt in Linux. The XCB version of Qt display
      has been found to avoid this issue. This can be done by first installing the ``xcb-cursor0`` library:

      .. code-block:: console

         sudo apt install libxcb-cursor0
       
      Then, add the following environmental variable to force Qt to use XCB display:

      .. code-block:: console

         export QT_QPA_PLATFORM=xcb
      
      To make this environmental variable change permanent, add it to your ``bashrc`` file using ``nano ~/.bashrc``
      and ``source ~/.bashrc``.
