Examples
================================================

1a. Antiviral Drugs Repurposing for SARS-CoV2 3CLPro, using One Line.**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a new target sequence (e.g. SARS-CoV2 3CL Protease), 
retrieve a list of repurposing drugs from a curated drug library of 81 antiviral drugs. 
The Binding Score is the Kd values. 
Results aggregated from five pretrained model on BindingDB dataset!

.. code-block:: python


	from DeepPurpose import oneliner
	oneliner.repurpose(*load_SARS_CoV2_Protease_3CL(), *load_antiviral_drugs())


