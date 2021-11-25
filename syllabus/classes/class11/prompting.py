from transformers import pipeline, set_seed, AutoTokenizer, AutoModelWithLMHead
set_seed(42)

 
tokenizer = AutoTokenizer.from_pretrained("flax-community/dansk-gpt-wiki")

model = AutoModelWithLMHead.from_pretrained("flax-community/dansk-gpt-wiki")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)  # Note that a larger model is likely to produce a better result.

# new york times article
TEXT = """Hvis det står til regeringen, så skal du igen trække mundbindet på, hvis du skal ned efter en liter mælk. Det gælder også, hvis du er den, som står bag kassen og sælger mælken.

Det sagde regeringen i går på et pressemøde, hvor den præsenterede en række af restriktioner, som skal dæmpe coronasmitten.

Men det bør ikke være nødvendigt, når arbejdsgivere kan kræve et grønt coronapas af deres ansatte, siger flere partier.

- Jeg synes, man bliver nødt til at finde en ordning for de mennesker, der arbejder i en butik en hel arbejdsdag, om de skal bære mundbind, siger Venstres sundhedsordfører, Martin Geertsen.

Partierne skal i dag mødes med regeringen i Epidemiudvalget, hvor de skal diskutere og beslutte, hvilke coronarestriktioner som skal finde vejen tilbage i vores hverdag.

Som udgangspunkt går SF ind for at følge anbefalingerne fra myndighederne, siger sundhedordfører for partiet Kirsten Normann Andersen. SF synes dog, det giver mening at se på, hvorvidt der også skal kræves mundbind i for eksempel detailhandlen, hvis arbejdsgiveren samtidig forlanger coronapas.

I Dansk Folkeparti er man helt imod brug af mundbind i supermarkeder - både for de handlende og for de ansatte.

- Vi synes, vi bliver nødt til at kigge på det her, så det ikke bliver så vidt, som det er foreslået fra regeringen. Vi kan ikke se fornuften i, at folk skal bruge mundbind i supermarkedet, siger Liselott Blixt, der er sundhedsordfører for DF.

Hun mener også, at det ikke det giver nogen mening, at butiksansatte skal bære mundbind, når deres arbejdsgiver samtidig kan kræve at se et coronapas.
"""

prompt = "Hun mener derudover"

output = generator(TEXT + prompt, 
                   max_length=630,  # max_length should be adaptable to model input size
                   num_return_sequences=1
                   )

print(output[0]['generated_text'])