# Multithreaded OLS Regression Shell

### Collaborators

 faheemmu@grinnell.edu  Muhammad Faheem
 khanalsa@grinnell.edu Sauryanshu Khanal

### Implementation

We will conduct an OLS regression with a .csv file as an input and we will output the results of running the regression in a text file. The output will include coefficient, variance, standard errors and p-values with the specific regression model that the user specifies in the shell. 
More information: https://www.statology.org/multiple-linear-regression/
Our implementation will only contain 3 numerical variables (and hence three variable datasets): a Y-var and two X Vars.

### Shell Implementation

We have our own shell that has 5 commands:
```
load anyFileName.csv // this command will load specified .csv file (should be present in repository)
view                 // this command will show you the data from the .csv file you loaded
reg var0 var1 var2   // this will form regression model with y as dependent variable and x1 and x2 as independent 
export output.txt    // this will create a text file of specified name and export the regression to that
exit                 // this will exit the shell 
```

### Example run of code:

1. Import the GNU Science Library. We use this library to import a T-Distribution that does a two-tailed test to calculate coefficient p-Values.
```
sudo apt-get install libgsl-dev
```
2. Run Make command 
```
make
```
4. Run ./shell to load shell
```
./shell
``` 

##### Inside the shell run command:

```
load sample_data.csv   		// Load the dataset
view              			// Print loaded dataset 
reg height weight age       // Run the regression and display its output
```
The output should be displayed as follows: 
![enter image description here](https://imgur.com/0HgUX3zl.png)

```
reg weight height age       // Run a different regression and display its output
```
![enter image description here](https://imgur.com/N3sSPphl.png)

```
export output.txt 		// Put the regression data in output.txt 
```
Note: it will put the regression results of most recent model (reg x1 x2 y in our case) 
```
exit              // Exit the terminal 
```



### Notes
1. Inside the shell, you can always load new datasets as you see fit to run new regressions.
2. We have appropriate error handling inside the shell, and also while doing parallel computation, so that the shell exits in the face of unexpected errors.
### References:
1. https://www.gnu.org/software/gsl/
2. https://www.statology.org/multiple-linear-regression/
3. https://www.statology.org/multiple-linear-regression-by-hand/
4. Econometrics Textbook:https://global.oup.com/ushe/product/real-econometrics-9780190857462?cc=us&lang=en&












