// Sample dataset with deliberate flaws (a duplicate row, a blank answer)
export const DEMO =
  'instruction,response\n"Summarize: The cat sat.","A cat sat."\n"Translate hello to French","Bonjour"\n"Translate hello to French","Bonjour"\n"Name a prime number",""\n"What is 2+2?","4"\n"Capital of France?","Paris"';

if (typeof window !== 'undefined') {
  window.DEMO = DEMO;
}
