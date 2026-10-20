import { expect, it, vi } from 'vitest';
import { subscribeMediaPaused } from './waveform-player';

it('tracks the native media element play and pause state for its button', () => {
  const media = Object.assign(new EventTarget(), { paused: true });
  const onPaused = vi.fn();
  const unsubscribe = subscribeMediaPaused(media as HTMLMediaElement, onPaused);

  media.paused = false;
  media.dispatchEvent(new Event('playing'));
  media.paused = true;
  media.dispatchEvent(new Event('pause'));

  expect(onPaused.mock.calls.map(([paused]) => paused)).toEqual([true, false, true]);

  unsubscribe();
  media.paused = false;
  media.dispatchEvent(new Event('playing'));
  expect(onPaused).toHaveBeenCalledTimes(3);
});
