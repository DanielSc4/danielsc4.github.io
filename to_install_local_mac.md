The original mac version of ruby should not be used
Install rbenv as follow:
``` bash
brew install rbenv 
rbenv install 3.1.2
rbenv local 3.1.2
```
Then reboot the terminal and check which ruby installation is loaded with: `which ruby`.
The output should not be `/usr/bin/ruby` but something like `/Users/daniel/.rubies/ruby-3.3.0/bin/ruby`

Then follow the instructions in INSTALL.MD
``` bash
bundle install
# assuming pip is your Python package manager
pip install jupyter
bundle exec jekyll serve --lsi
```
