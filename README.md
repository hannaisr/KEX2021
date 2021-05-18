# KEX2021
![Image of fingerprint](https://images.theconversation.com/files/261822/original/file-20190304-110110-1tgw1we.jpg?ixlib=rb-1.1.0&amp;rect=8%2C735%2C5633%2C2816&amp)
<i>Image from <a href="https://theconversation.com/fingerprint-and-face-scanners-arent-as-secure-as-we-think-they-are-112414" title="Fingerprint and face scanners arent as secure as we think they are">The Conversation</a> (interesting article).</i>

### The files in this repository
<ul>
  <li><b>datastorage_DataFrame.ipynb</b> - Code for storing images as pickle file.</li>
  <li><b>image_rotation.ipynb</b> - Code for extending the dataset by creating multiple copies of each image, altered by Gaussian distributed rotations and shifts.</li>
  <li><b>10_first_ppl_100_rots.pkl</b> - A pickle file with fingerprints from the first ten people in the original SOCOF dataset, extended to include 100 altered copies of each original image.</li>
  <li><b>Simple Random Forest.ipynb</b> - Code for single-stage identification, including algorithms for cross validation and determining how the number of altered copies affect accuracy.</li>
  <li><b>mnist_test.ipynb</b> - Code for testing how well the dataset extension works.</li>
  <li><b>two_stage_rf.ipynb</b> - Code for two-stage identification.</li>
</ul>

### Overleaf
<a href="https://www.overleaf.com/2226766734gdpzthcqxbvg" title="KEX 2021 - Report">Report</a><br>
<a href="https://www.overleaf.com/3585548193prqsgrjjfbnx" title="Arbetsplan KEX2021">Work plan</a>

## Tips and tricks
#### File conversion
To convert .ipynb files to .py, write <br>
```
$ jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
```
in command prompt, with [YOUR_NOTEBOOK] being the name of the notebook.

#### Datasets
Many different datasets have been suggested to use while developing the programmes, but scikit has many convenient datasets included and they are simpler to use and do not require any downloads. The datasets can be collected with the commands

<table class="longtable docutils align-default">
<tbody>
<tr class="row-odd"><td><a class="reference internal" href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston" title="sklearn.datasets.load_boston"><code class="xref py py-obj docutils literal notranslate"><span class="pre">load_boston</span></code></a>(*[, return_X_y])</td>
<td>Load and return the boston house-prices dataset (regression).</td>
</tr>
<tr class="row-even"><td><a class="reference internal" href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris" title="sklearn.datasets.load_iris"><code class="xref py py-obj docutils literal notranslate"><span class="pre">load_iris</span></code></a>(*[, return_X_y, as_frame])</td>
<td>Load and return the iris dataset (classification).</td>
</tr>
<tr class="row-odd"><td><a class="reference internal" href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes" title="sklearn.datasets.load_diabetes"><code class="xref py py-obj docutils literal notranslate"><span class="pre">load_diabetes</span></code></a>(*[, return_X_y, as_frame])</td>
<td>Load and return the diabetes dataset (regression).</td>
</tr>
<tr class="row-even"><td><a class="reference internal" href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits" title="sklearn.datasets.load_digits"><code class="xref py py-obj docutils literal notranslate"><span class="pre">load_digits</span></code></a>(*[, n_class, return_X_y, as_frame])</td>
<td>Load and return the digits dataset (classification).</td>
</tr>
<tr class="row-odd"><td><a class="reference internal" href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud" title="sklearn.datasets.load_linnerud"><code class="xref py py-obj docutils literal notranslate"><span class="pre">load_linnerud</span></code></a>(*[, return_X_y, as_frame])</td>
<td>Load and return the physical excercise linnerud dataset.</td>
</tr>
<tr class="row-even"><td><a class="reference internal" href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine" title="sklearn.datasets.load_wine"><code class="xref py py-obj docutils literal notranslate"><span class="pre">load_wine</span></code></a>(*[, return_X_y, as_frame])</td>
<td>Load and return the wine dataset (classification).</td>
</tr>
<tr class="row-odd"><td><a class="reference internal" href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer" title="sklearn.datasets.load_breast_cancer"><code class="xref py py-obj docutils literal notranslate"><span class="pre">load_breast_cancer</span></code></a>(*[, return_X_y, as_frame])</td>
<td>Load and return the breast cancer wisconsin dataset (classification).</td>
</tr>
</tbody>
</table>

For more details, follow <a href="https://scikit-learn.org/stable/datasets/toy_dataset.html#toy-datasets" title="Toy datasets">this link</a>.
