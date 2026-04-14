/**
 * Built-in sample corpus for the auto-test mode.
 *
 * 20 paragraphs spanning distinct topics so that a semantic query derived
 * from the first sentence of each paragraph reliably retrieves that paragraph.
 * Each entry carries the paragraph text and a focused query string.
 */

export interface CorpusEntry {
  id: number;
  text: string;
  /** A specific phrase from the paragraph used as the recall query. */
  query: string;
}

export const SAMPLE_CORPUS: CorpusEntry[] = [
  {
    id: 0,
    text: `Quantum computing exploits the principles of quantum mechanics to process information in ways that classical computers cannot. Unlike classical bits, which are always in state 0 or 1, quantum bits (qubits) can exist in superposition — simultaneously representing both states. Entanglement further links qubits so that the state of one instantly influences another regardless of distance. These properties enable quantum computers to solve certain problems, such as factoring large integers or simulating molecular interactions, exponentially faster than any known classical algorithm.`,
    query: 'quantum bits superposition entanglement computing',
  },
  {
    id: 1,
    text: `Ancient Rome was one of the largest and most powerful empires in history, spanning from Britain in the north to North Africa in the south and from Spain in the west to Mesopotamia in the east. At its peak under Emperor Trajan in 117 AD, the Roman Empire covered approximately 5 million square kilometres and governed around 70 million people. Roman engineering achievements — aqueducts, roads, the Colosseum, the Pantheon — remain visible today and influenced Western civilization profoundly.`,
    query: 'Roman Empire aqueducts Trajan Colosseum engineering',
  },
  {
    id: 2,
    text: `Photosynthesis is the biological process by which plants, algae, and some bacteria convert sunlight into chemical energy stored as glucose. In the light-dependent reactions occurring in the thylakoid membranes, chlorophyll absorbs photons and splits water molecules, releasing oxygen as a byproduct. The resulting ATP and NADPH power the Calvin cycle in the stroma, where carbon dioxide is fixed into three-carbon sugars. Photosynthesis produces virtually all oxygen in Earth's atmosphere and forms the base of nearly every food chain.`,
    query: 'photosynthesis chlorophyll Calvin cycle oxygen carbon dioxide',
  },
  {
    id: 3,
    text: `The Renaissance was a cultural and intellectual movement that began in Italy during the 14th century and gradually spread across Europe, marking the transition from the Middle Ages to modernity. Artists like Leonardo da Vinci, Michelangelo, and Raphael pioneered new techniques including perspective, sfumato, and chiaroscuro to achieve unprecedented realism. Humanist scholars rediscovered classical Greek and Roman texts, placing humanity at the centre of philosophical inquiry. The invention of the printing press by Gutenberg in 1440 dramatically accelerated the dissemination of Renaissance ideas.`,
    query: 'Renaissance Leonardo da Vinci perspective Michelangelo humanism',
  },
  {
    id: 4,
    text: `Black holes are regions of spacetime where gravity is so strong that nothing — not even light — can escape once it crosses the event horizon. They form when massive stars collapse under their own gravity at the end of their life cycle in a supernova explosion. The singularity at the centre is a point of infinite density where current physical laws break down. Stephen Hawking predicted that black holes slowly evaporate by emitting radiation, now called Hawking radiation, due to quantum effects near the event horizon.`,
    query: 'black hole event horizon singularity Hawking radiation',
  },
  {
    id: 5,
    text: `The French Revolution, beginning in 1789, was a period of radical political and societal transformation in France that overthrew the monarchy and established a republic. Fuelled by Enlightenment ideas, economic hardship, and social inequality between the Three Estates, the revolution gave rise to the Declaration of the Rights of Man and the Citizen. The Reign of Terror under Robespierre saw thousands guillotined. Napoleon Bonaparte eventually rose from the revolutionary chaos to become Emperor, spreading revolutionary ideals across Europe through his military campaigns.`,
    query: 'French Revolution Robespierre guillotine Napoleon Bonaparte 1789',
  },
  {
    id: 6,
    text: `Machine learning is a subfield of artificial intelligence in which systems learn from data to improve their performance on tasks without being explicitly programmed. Supervised learning trains models on labelled examples; unsupervised learning finds hidden patterns in unlabelled data; reinforcement learning trains agents through reward signals. Deep neural networks, composed of many layers of interconnected nodes, have achieved superhuman performance in image recognition, natural language processing, and game-playing since the deep learning breakthrough of 2012.`,
    query: 'machine learning supervised neural networks deep learning AI',
  },
  {
    id: 7,
    text: `The structure of DNA was elucidated by James Watson and Francis Crick in 1953, building on X-ray crystallography data from Rosalind Franklin. DNA is a double helix composed of two antiparallel strands of nucleotides, where adenine pairs with thymine and guanine pairs with cytosine through hydrogen bonds. This complementary base-pairing mechanism enables faithful replication of genetic information during cell division. The sequence of base pairs encodes the genetic instructions for building and operating all known living organisms.`,
    query: 'DNA double helix Watson Crick base pairs nucleotides',
  },
  {
    id: 8,
    text: `The Amazon rainforest is the world's largest tropical rainforest, covering over 5.5 million square kilometres across nine South American countries, with Brazil containing about 60% of its area. It is home to an estimated 10% of all species on Earth, including more than 40,000 plant species, 1,300 bird species, and 3,000 freshwater fish species. The Amazon Basin produces around 20% of the world's oxygen and plays a critical role in regulating global climate patterns through evapotranspiration and carbon sequestration.`,
    query: 'Amazon rainforest biodiversity evapotranspiration carbon oxygen',
  },
  {
    id: 9,
    text: `Medieval castles served multiple functions as fortified residences, administrative centres, and symbols of feudal power. The development of the concentric castle design — with multiple rings of defensive walls — reached its peak during the Crusades. Key architectural elements included the keep, moat, drawbridge, portcullis, battlements, and arrow loops. Advances in siege warfare, including trebuchets, undermining, and eventually gunpowder artillery, gradually rendered traditional stone castles obsolete by the late medieval period.`,
    query: 'medieval castle keep moat trebuchet siege warfare portcullis',
  },
  {
    id: 10,
    text: `Ocean currents are continuous, directed movements of seawater driven by differences in temperature, salinity, wind, and the Earth's rotation (Coriolis effect). The global thermohaline circulation — sometimes called the ocean conveyor belt — transports warm surface water from the tropics toward the poles, where it cools, becomes denser, and sinks, driving a slow deep-water return flow. This circulation distributes heat around the planet, moderating climates in coastal regions. A slowdown in the Atlantic Meridional Overturning Circulation is considered one of the major risks of climate change.`,
    query: 'ocean thermohaline circulation conveyor belt Coriolis salinity',
  },
  {
    id: 11,
    text: `The Industrial Revolution, which began in Britain around 1760, transformed manufacturing from cottage industries to mechanised factory production. The invention of the steam engine by James Watt enabled factories to be located away from rivers, and railways rapidly connected industrial centres. Urbanisation accelerated dramatically as workers migrated to cities. Child labour, dangerous working conditions, and pollution were widespread, eventually provoking social reform movements, trade unionism, and landmark legislation like the Factory Acts.`,
    query: 'Industrial Revolution steam engine James Watt factories railways',
  },
  {
    id: 12,
    text: `The human brain contains approximately 86 billion neurons, each forming up to 10,000 synaptic connections with other neurons, resulting in roughly 100 trillion synapses. Neurotransmitters such as dopamine, serotonin, glutamate, and GABA mediate communication across synaptic clefts. Neuroplasticity — the brain's ability to reorganise its structure and function in response to experience — underlies learning and memory. Disorders such as Alzheimer's disease involve progressive loss of synaptic connections and neuronal death, especially in the hippocampus and cortex.`,
    query: 'neurons synapses neurotransmitters dopamine neuroplasticity brain',
  },
  {
    id: 13,
    text: `Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused since the mid-20th century by human activities emitting greenhouse gases such as CO₂, methane, and nitrous oxide. The burning of fossil fuels, deforestation, and industrial agriculture are the main contributors. Rising temperatures are causing polar ice melt, sea-level rise, more frequent extreme weather events, and shifts in ecosystems. The Paris Agreement of 2015 committed nations to limiting warming to 1.5°C above pre-industrial levels.`,
    query: 'climate change CO2 greenhouse gases Paris Agreement fossil fuels',
  },
  {
    id: 14,
    text: `Ancient Egypt was one of the earliest and longest-lasting civilisations, flourishing along the Nile River for over 3,000 years from around 3100 BCE. The pharaohs ruled as god-kings, overseeing the construction of monumental architecture including the pyramids of Giza and the Sphinx. Egyptian hieroglyphics, one of the earliest writing systems, recorded religious texts, administrative records, and historical accounts on papyrus and stone. The decipherment of the Rosetta Stone by Champollion in 1822 unlocked our understanding of this ancient script.`,
    query: 'ancient Egypt pharaoh pyramids hieroglyphics Rosetta Stone Nile',
  },
  {
    id: 15,
    text: `Einstein's theory of special relativity, published in 1905, revolutionised physics by postulating that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is constant regardless of the motion of the observer or source. This led to the famous equation E=mc², expressing the equivalence of mass and energy. Time dilation and length contraction are measurable consequences at velocities approaching the speed of light. General relativity, published in 1915, extended these principles to gravity and accelerating reference frames.`,
    query: 'Einstein relativity E=mc2 time dilation speed of light',
  },
  {
    id: 16,
    text: `Coral reefs are among the most biodiverse ecosystems on Earth, occupying less than 0.1% of the ocean floor yet supporting about 25% of all marine species. Built by coral polyps that secrete calcium carbonate skeletons, reefs provide habitat, feeding grounds, and nurseries for thousands of fish species. Ocean warming causes coral bleaching — the expulsion of symbiotic algae (zooxanthellae) — leading to mass die-offs. The Great Barrier Reef has experienced multiple severe bleaching events since 2016, driven by record sea surface temperatures.`,
    query: 'coral reef bleaching zooxanthellae Great Barrier Reef biodiversity',
  },
  {
    id: 17,
    text: `The Silk Road was an ancient network of trade routes connecting China to the Mediterranean world via Central Asia, active from approximately 130 BCE to 1450 CE. Silk, spices, glass, paper, and gunpowder were among the commodities exchanged, but the routes were equally important for the transmission of religion, art, disease, and technology. Buddhism spread from India to China along these routes, while the Black Death may have spread westward along them in the 14th century. Marco Polo's travels in the 13th century brought knowledge of East Asia to European audiences.`,
    query: 'Silk Road trade China Mediterranean Buddhism Marco Polo spices',
  },
  {
    id: 18,
    text: `Cryptography is the science of securing communication through mathematical techniques that render messages unreadable to unauthorised parties. Classical ciphers like Caesar's substitution and Vigenère's polyalphabetic cipher were eventually broken by frequency analysis. Modern cryptography relies on computational hardness: RSA encryption depends on the difficulty of factoring large prime numbers, while elliptic-curve cryptography offers equivalent security with shorter keys. Public-key infrastructure (PKI) enables secure HTTPS connections on the internet by allowing parties who have never met to exchange encrypted messages.`,
    query: 'cryptography RSA encryption prime numbers elliptic curve HTTPS',
  },
  {
    id: 19,
    text: `Volcanoes are openings in Earth's crust through which molten rock (magma), volcanic ash, and gases escape from below the surface. Shield volcanoes, like those in Hawaii, produce low-viscosity basaltic lava flows and rarely explode violently. Stratovolcanoes, such as Mount Vesuvius and Mount Pinatubo, can produce catastrophic Plinian eruptions ejecting pyroclastic flows and ash clouds that cool global temperatures. The 1991 eruption of Pinatubo injected so much sulfur dioxide into the stratosphere that it reduced global temperatures by 0.5°C for two years.`,
    query: 'volcano eruption magma pyroclastic Pinatubo stratosphere ash',
  },
];
