import heyoka as hy
print(f"Heyoka Version: {hy.__version__}")
print("Attributes:")
print([x for x in dir(hy) if 'par' in x or 'make' in x])
