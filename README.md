# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
We use pytest and pytest-cov plugin for testing.

---

##### Usage:
in tests directory:
- `pytest` - run all tests
- `pytest -m models` - run tests for all available models config files
- `pytest -m 'not models'` - run all tests without models testing
- `pytest -m seresnext` - run tests for seresnext model
- `pytest -m resnet` - run tests for resnet model
- `pytest -m densenet` - run tests for densenet model
- `pytest -m espnet` - run tests for espnet model
- `pytest -m validator` - run tests for config files validation
- `pytest -m metric_utils` - run metric utils tests
- `pytest -m losses` - run losses utils tests

---

After passing tests you can find the test report inside tests/htmlcov. To view report open index.html file (each
 tests run will override the existing report).


# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://www.visualstudio.com/en-us/docs/git/create-a-readme). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)