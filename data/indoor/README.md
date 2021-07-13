### Benchmark dataset에 대한 description - New visit에 대한 재방문 예측 세팅

실제 매장에서의 고객의 방문 데이터는 스트림 데이터이다.
스트림 데이터는 시간의 전후 관계를 가지고 있기 때문에, 매장에 방문한 고객의 재방문을 실시간 예측할 때에는, 이전에 수집된 모든 데이터를 train data로 활용할 수가 있을 것이다. 하지만 실제 현장에서 스트림 데이터를 활용한다고 해서, 모델을 매번 업데이트하지는 않을 것이며, 일반적으로는 dump 데이터를 활용하여 학습시킨 모델을 일정 기간 동안 활용할 것이다.

Benchmark 데이터셋을 만들면서, 미묘한 문제를 고려하여야만 했다. 
우선 스트림 데이터 상황에서 연속적인 테스트가 가능하게끔 해야 하는 가에 대한 문제이다.
이렇게 테스팅 플렛폼을 만들 수 있다면, online test와 완전히 같은 세팅을 가져올 수 있다는 장점이 있는 반면, 다음과 같은 커다란 단점이 있다. 매 로그가 발생할 때마다, Train data가 늘어나는 상황인데 해당 유저의 직전 visit의 label을 update해 주어야 하며, 해당 유저에 대한 feature값 역시 업데이트를 하고 prediction을 수행해야 최적의 성능을 낼 수 있을 것이다. 이렇게 까다로운 세팅은 현업에서 필요에 의해 구현할 수 있겠지만, 재방문을 연구의 저변을 넓히기 위하여 공개하는 benchmark dataset의 의의에는 적절하지 않다. 

따라서 하나의 timestamp를 정해두고, 그 전에 발생한 방문과 label을 training data에, 그 이후에 발생한 방문과 label을 test data로 자른 형태의 재방문 연구용 데이터셋을 공개하는 게 적합하다고 판단하였다. 이렇게 세팅할 경우 특정 시점에서의 재방문 예측을 전제로 연구를 수행할 수가 있다고 판단하였다. 다만 시간 순서로 쪼개는 연구의 특성상 몇 가지 짚고 넘어가야 할 점이 있는데, 다음과 같다.

우선은 Test timeframe에 여러 번 방문한 고객의 경우 첫 번째 visit에 해당되는 로그만을 예측 대상으로 남겨놓기로 하였다. 해당 고객의 모든 visit들을 test data에 포함시킨다면, 정상적인 머신 러닝 모델을 활용하여 해당 visit들의 재방문을 예측하는 대신, 고객의 identifier가 여러 번 존재한다는 점만을 비정상적으로 활용하여 test data에 여러 번 방문한 고객들의 재방문을 예측할 수 있을 거라 판단하였기 때문이다. training timeframe에 방문한 횟수는 이 문제와는 전혀 상관이 없으며, training timeframe에 방문 없이 test 데이터에 처음 방문한 고객의 경우에도 당연히 예측 대상에 포함된다.  

training timeframe에 방문하고 test timeframe에 다시 방문한 고객의 경우, training timeframe에 해당되는 마지막 방문의 label을 어떻게 처리하여야 하느냐에 대한 문제이다. 물론 time t라는 train/test cutoff point에 근거하여 time t 전의 데이터만 활용하여 training dataset의 labeling을 해 주어야 하는 것이 전혀 leakage가 발생하지 않고 합당하다. 




~~
하지만 leakage를 전혀 없애는 이 솔루션을 채택하면 benchmark 데이터셋에서 엄청난 information loss를 가져오게 된다. 애초에 revisit class imbalance가 굉장히 심한 데이터인데, test timeframe에 재방문한 training timeframe의 고객들의 재방문 여부를 no라고 표시해두게 되면, class imbalance가 더욱 심해지게 된다. 이를 방지하고자, test timeframe에 재방문한 고객의 경우 training data의 마지막 방문의 revisit intention label은 True로 표시하고, 해당하는 revisit interval을 표기하기로 한다. 실제 스트림 데이터 상황에서도 test data에 해당되는 visit의 prediction을 진행할 시점에는 직전 방문의 재방문 여부가 True임을 알 수 있기 때문에, 이 세팅이 적절하다고 생각된다.

따라서 최종적으로 만들어진 dataset의 경우 training timeframe에 방문한 고객의 revisit intention, revisit interval 라벨은 test timeframe에 다시 방문하지 않는 경우를 제외하고는 모두 제공하는 정책을 취했다고 얘기할 수 있다.

To-do: 매장 A, B는 기존 연구에서 사용된 매장 두개를 어떤 식으로 샘플링했는지 마찬가지로 소개할 필요가 있다.
~~