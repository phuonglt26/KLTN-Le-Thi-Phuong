import setuptools

# with open("ai_project_template/README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="bert_entity_extractor",
    version="1.0",
    author="hihi",
    author_email="hihi@gmail.com",
    description="AI models for entity Extractor",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # install_requires=["tensorflow>=1.8.0", "keras>=2.2.0"],
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={"bert_entity_extractor": []},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
